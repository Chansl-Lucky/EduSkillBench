"""Process-local reliability fixes; never mutate pinned BenchFlow source files.

Agent prompts, tools, decoding and 600-second execution budget are unchanged.
Only setup scheduling/health, subprocess cleanup and pre-response transport
retries change. Loaded exclusively by the separately versioned recovery run.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
from pathlib import Path
import time

PHASE = contextvars.ContextVar('glm305_lifecycle_phase', default='unknown')
STARTUP_COMMAND_FLOOR = 45
HEALTH_TIMEOUT = 240
CONNECT_RETRIES = 3
_INSTALLED = False


class StageFailure(RuntimeError):
    def __init__(self, phase, exc):
        self.phase = phase
        self.exception_type = type(exc).__name__
        super().__init__(f'{phase}: {self.exception_type}: {str(exc) or "(empty exception message)"}')


class DockerCommandTimeout(RuntimeError):
    pass


async def reap_process(process, communicate):
    """A raced process exit must not replace the original timeout exception."""
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
    try:
        return await asyncio.wait_for(asyncio.shield(communicate), 5)
    except TimeoutError:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        try:
            return await asyncio.wait_for(asyncio.shield(communicate), 5)
        except TimeoutError:
            communicate.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await communicate
            return b'', b''


async def docker_compose_command(self, command, check=True, timeout_sec=None):
    from benchflow.sandbox.docker import _sanitize_docker_compose_project_name
    from benchflow.sandbox._base import ExecResult
    argv = ['docker', 'compose', '--project-name',
            _sanitize_docker_compose_project_name(self.session_id),
            '--project-directory', str(self.environment_dir.resolve().absolute())]
    for path in self._docker_compose_paths:
        argv.extend(['-f', str(path.resolve().absolute())])
    argv.extend(command)
    phase = PHASE.get()
    # Do not increase time allowed for agent-authored tool commands.
    effective = timeout_sec
    if phase in ('setup', 'environment_start', 'agent_install', 'agent_connect') and timeout_sec:
        effective = max(timeout_sec, STARTUP_COMMAND_FLOOR)
    process = await asyncio.create_subprocess_exec(
        *argv, env=self._docker_compose_env(), stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    communication = asyncio.create_task(process.communicate())
    try:
        output, stderr = await asyncio.wait_for(asyncio.shield(communication), effective) if effective else await communication
    except TimeoutError:
        await reap_process(process, communication)
        # No argv/payload here: installation commands can contain credentials.
        raise DockerCommandTimeout(f'Docker compose command timed out during {phase} after {effective}s') from None
    except asyncio.CancelledError:
        await reap_process(process, communication)
        raise
    result = ExecResult(stdout=output.decode(errors='replace') if output else None,
                        stderr=stderr.decode(errors='replace') if stderr else None,
                        return_code=process.returncode or 0)
    if check and result.return_code != 0:
        # Avoid putting entire credential-bearing installation commands in errors.
        raise RuntimeError(f'Docker compose command failed during {phase}; return_code={result.return_code}')
    return result


async def poll_host_health(process, deadline_s=HEALTH_TIMEOUT):
    import httpx
    started = time.monotonic()
    last = 'no response'
    # Local readiness must never use inherited HTTP_PROXY; keep one client.
    async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
        while time.monotonic() - started < deadline_s:
            if process.process.poll() is not None:
                raise RuntimeError('LiteLLM exited before readiness')
            for path in ('/health/liveliness', '/health'):
                try:
                    r = await client.get(process.endpoint.local_base_url + path)
                    if r.status_code == 200:
                        return
                    last = f'HTTP {r.status_code}'
                except httpx.HTTPError as exc:
                    last = type(exc).__name__
            await asyncio.sleep(.5)
    raise RuntimeError(f'LiteLLM healthcheck timed out after {deadline_s}s: {last}')


def install_runtime_fixes():
    global _INSTALLED
    if _INSTALLED:
        return
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.providers import litellm_runtime
    DockerSandbox._run_docker_compose_command = docker_compose_command
    litellm_runtime._poll_host_health = poll_host_health
    _INSTALLED = True


def recovery_rollout_class(harness, startup_semaphore, save):
    parent = harness.audited_rollout_class()

    class RecoveryRollout(parent):
        _startup_held = False
        recovery_phase = 'created'
        recovery_exception = None

        def persist_phase(self):
            if self._rollout_dir is not None:
                save(self._require_rollout_dir() / 'capture/recovery_phase.json', {
                    'phase': self.recovery_phase, 'exception': self.recovery_exception,
                    'startup_held': self._startup_held})

        async def stage(self, name, fn, *args, **kwargs):
            self.recovery_phase = name
            token = PHASE.set(name)
            self.persist_phase()
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                self.recovery_exception = {'phase': name, 'type': type(exc).__name__,
                                           'message': str(exc)[:400]}
                self.persist_phase()
                if name == 'agent_execution':
                    raise
                raise StageFailure(name, exc) from exc
            finally:
                PHASE.reset(token)

        async def setup(self):
            await startup_semaphore.acquire()
            self._startup_held = True
            return await self.stage('setup', super().setup)

        async def start(self):
            return await self.stage('environment_start', super().start)

        async def install_agent(self):
            return await self.stage('agent_install', super().install_agent)

        async def connect_as(self, role):
            try:
                return await self.stage('agent_connect', super().connect_as, role)
            finally:
                self.release_startup()

        async def connect(self):
            try:
                return await self.stage('agent_connect', super().connect)
            finally:
                self.release_startup()

        async def execute(self, *args, **kwargs):
            return await self.stage('agent_execution', super().execute, *args, **kwargs)

        def release_startup(self):
            if self._startup_held:
                startup_semaphore.release()
                self._startup_held = False

        async def cleanup(self):
            self.release_startup()
            token = PHASE.set('cleanup')
            self.persist_phase()
            try:
                return await super().cleanup()
            finally:
                PHASE.reset(token)

    return RecoveryRollout


async def open_upstream(client, api, headers, payload, log, request_id, retry_limit=CONNECT_RETRIES):
    """Retry only transport failures BEFORE response headers. No stream replay."""
    import httpx
    for attempt in range(retry_limit):
        started = time.monotonic()
        try:
            response = await client.send(client.build_request('POST', api, headers=headers, json=payload), stream=True)
            log({'request_id': request_id, 'attempt': attempt + 1,
                 'phase': 'headers_received', 'status': response.status_code,
                 'seconds': round(time.monotonic() - started, 3)})
            return response
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            log({'request_id': request_id, 'attempt': attempt + 1,
                 'phase': 'pre_response_connect_failure', 'type': type(exc).__name__,
                 'seconds': round(time.monotonic() - started, 3)})
            if attempt + 1 == retry_limit:
                raise
            await asyncio.sleep(min(2 ** attempt, 4))
