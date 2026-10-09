"""Portable Docker/OpenCode evaluation with task-scoped native Skill access."""
from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from . import harness, runtime
from .upstream import judge, source_protocol as native


def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
read, save = harness.read, harness.save


def parse_data(row):
    value = row.get('data', {})
    return json.loads(value) if isinstance(value, str) else value


def final_answer(db):
    roots = {r['id'] for r in db.get('session', []) if not r.get('parent_id')}
    roots = {m['session_id'] for m in db.get('message', []) if m.get('session_id') in roots and parse_data(m).get('role') == 'user'}
    if len(roots) != 1: return '', {'terminal': False, 'reason': 'missing_or_ambiguous_parent'}
    root = next(iter(roots))
    messages = [(m, parse_data(m)) for m in db.get('message', []) if m.get('session_id') == root and parse_data(m).get('role') == 'assistant']
    messages.sort(key=lambda x: (x[0].get('time_created', 0), x[0]['id']))
    if not messages: return '', {'terminal': False, 'reason': 'missing_assistant'}
    row, body = messages[-1]
    parts = [p for p in db.get('part', []) if p.get('message_id') == row['id'] and parse_data(p).get('type') == 'text' and not parse_data(p).get('synthetic')]
    parts.sort(key=lambda p: (p.get('time_created', 0), p['id']))
    answer = '\n'.join(parse_data(p).get('text', '') for p in parts).strip()
    return answer, {'terminal': bool(answer and body.get('finish') in ('stop', 'end_turn')),
                    'session_id': root, 'message_id': row['id'], 'finish': body.get('finish')}


def build_task(case, condition, out, skills, image, candidates):
    path = out / 'tasks' / condition / case['task_id']
    prompt = case['context'] + '\n\n' + case['user_prompt'] + '\n'
    bank = candidates.get(case['task_id'], [case['task_id'].split('__')[0]])
    if not bank or len(set(bank)) != len(bank): raise ValueError('Invalid candidate bank')
    if any('/' in x or '\\' in x or x in ('.', '..') for x in bank): raise ValueError('Invalid Skill ID')
    if path.exists():
        assert (path / 'instruction.md').read_text() == prompt
        if condition == 'with_skill':
            assert sorted(p.name for p in (path / 'environment/skills').iterdir()) == sorted(bank)
        return path
    env = path / 'environment'
    env.mkdir(parents=True)
    docker = f'FROM {image}\nRUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app /tests\n'
    setting = ''
    if condition == 'with_skill':
        for sid in bank:
            src = skills / sid
            if not (src / 'SKILL.md').is_file(): raise ValueError('Missing Skill: ' + sid)
            dst = env / 'skills' / sid
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('evals', '__pycache__', '.git'))
        for p in [env / 'skills', *(env / 'skills').rglob('*')]:
            p.chmod(0o755 if p.is_dir() or p.stat().st_mode & 0o111 else 0o644)
        docker += 'COPY skills/ /skills/\nRUN find /skills -type d -exec chmod 755 {} + && find /skills -type f -exec chmod a+r {} +\n'
        setting = '\nskills_dir="/skills"'
    (path / 'instruction.md').write_text(prompt)
    (env / 'Dockerfile').write_text(docker + 'WORKDIR /app\n')
    (path / 'task.toml').write_text('version="1.0"\n[metadata]\nauthor_name="EduSkillBench"\ndifficulty="medium"\ncategory="education"\n[agent]\ntimeout_sec=600\n[verifier]\ntimeout_sec=1900\n[environment]\ncpus=1\nmemory_mb=1024\nallow_internet=true' + setting + '\n')
    save(path / 'candidate_manifest.json', {'expected_skill': case['task_id'].split('__')[0],
        'candidate_skills': bank if condition == 'with_skill' else [], 'expected_is_mapping_not_certified_gold': True})
    return path


def route_audit(db, case, bank):
    calls, reads = [], []
    expected = case['task_id'].split('__')[0]
    for row in db.get('part', []):
        p = parse_data(row)
        if p.get('type') != 'tool': continue
        s = p.get('state', {}); inp = s.get('input', {})
        if p.get('tool') == 'skill': calls.append({'name': inp.get('name') if isinstance(inp, dict) else None, 'status': s.get('status')})
        if p.get('tool') in ('read', 'bash') and 'SKILL.md' in json.dumps(inp):
            reads.append({'tool': p['tool'], 'status': s.get('status'), 'expected_path_named': expected in json.dumps(inp)})
    return {'expected_skill': expected, 'candidate_skills': bank, 'skill_calls': calls,
            'skill_file_read_events': reads,
            'matched_skill_tool_completed': any(x['name'] == expected and x['status'] == 'completed' for x in calls),
            'matched_file_access_completed': any(x['expected_path_named'] and x['status'] == 'completed' for x in reads),
            'functional_adoption': 'not implied by access; requires content-level review'}


class Bridge:
    def __init__(self, args, token): self.args, self.token = args, token
    async def start(self):
        from aiohttp import web
        import httpx
        self.client = httpx.AsyncClient(trust_env=False, proxy=os.getenv('POLICY_PROXY'),
            timeout=httpx.Timeout(480, connect=15, read=60, pool=30),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=64))
        app = web.Application(client_max_size=8 * 1024 * 1024)
        app.router.add_route('*', '/{tail:.*}', self.handle)
        self.runner = web.AppRunner(app, access_log=None)
        self.closed = False
        await self.runner.setup()
        await web.TCPSite(self.runner, harness.gateway(), self.args.port).start()
    async def stop(self):
        if self.closed: return
        self.closed = True
        await self.runner.cleanup()
        await self.client.aclose()
    async def handle(self, request):
        from aiohttp import web
        a = self.args
        if request.headers.get('Authorization') != 'Bearer ' + self.token: return web.Response(status=401)
        if request.method == 'GET' and request.path == '/v1/models':
            return web.json_response({'object': 'list', 'data': [{'id': a.model, 'object': 'model'}]})
        if request.method != 'POST' or request.path != '/v1/chat/completions': return web.Response(status=404)
        payload = await request.json(); payload['model'] = a.model
        payload.update(json.loads(a.policy_options))
        cap = payload.pop('max_completion_tokens', payload.get('max_tokens', 10000))
        payload['max_tokens'] = min(int(cap or 10000), 10000)
        upstream = response = None
        try:
            async with asyncio.timeout(480):
                upstream = await runtime.open_upstream(self.client, a.policy_base_url.rstrip('/') + '/chat/completions',
                    {'Authorization': 'Bearer ' + os.environ['LLM_API_KEY']}, payload,
                    lambda x: harness.append(a.out / 'transport_attempts.jsonl', x), a.model)
                if upstream.status_code != 200 or not payload.get('stream'):
                    return web.Response(body=await upstream.aread(), status=upstream.status_code, content_type='application/json')
                response = web.StreamResponse(headers={'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache'})
                await response.prepare(request)
                pending, done = b'', False
                async for chunk in upstream.aiter_bytes():
                    pending += chunk
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        if line.startswith(b'data: ') and line[6:].strip() == b'[DONE]': done = True
                    await response.write(chunk)
                if not done: raise RuntimeError('Upstream stream ended without DONE')
                await response.write_eof()
                return response
        except Exception as exc:
            harness.append(a.out / 'provider_errors.jsonl', {'at': now(), 'error_type': type(exc).__name__})
            if response is not None and response.prepared:
                response.force_close()
                if request.transport: request.transport.close()
                return response
            return web.json_response({'error': {'message': type(exc).__name__}}, status=502)
        finally:
            if upstream is not None: await upstream.aclose()


def score_one(x):
    dest = Path(x['cell']); p = dest / 'score.json'
    if p.exists() and read(p)['status'] == 'evaluated': return read(p)
    case, answer = x['case'], x['answer']
    if case['suite'] == 'advisory':
        prompt = native.prompt(case, answer)
        schema = native.schema([c for c in case['criteria'] if c['id'] in case['applicable_ids']])
    else:
        prompt = judge.build_prompt({'question': case['context'] + '\n\n' + case['user_prompt'],
            'ground_truth': case['expected_output'], 'rubric': case['rubric'], 'score_metric': 'equal'}, answer)
        schema = judge.verdict_schema(case['rubric'])
    os.environ['ANTHROPIC_BASE_URL'] = x['judge_base_url']
    os.environ['ANTHROPIC_AUTH_TOKEN'] = os.environ.get('JUDGE_API_KEY', os.environ['LLM_API_KEY'])
    os.environ.pop('ANTHROPIC_API_KEY', None)
    proxy = os.getenv('JUDGE_PROXY')
    if proxy: judge.request_transport = lambda url: (urllib.request.build_opener(urllib.request.ProxyHandler({'https': proxy})).open, 'explicit_proxy')
    request = prompt
    for attempt in range(3):
        stamp = time.time_ns()
        try:
            raw, meta = judge.call_judge(request, 'deepseek-v4-pro', telemetry_path=dest / f'judge_trace_{stamp}.json',
                api_protocol='chat_completions', thinking_mode='disabled', max_output_tokens=10000,
                timeouts={'headers': 45., 'idle': 45., 'content_idle': 60., 'total': 120.}, output_schema=schema)
            (dest / f'judge_raw_{stamp}.txt').write_text(raw); save(dest / f'judge_response_{stamp}.json', meta)
            assert meta['response_model'] == 'deepseek-v4-pro-ga-260813', 'Judge identity changed'
            assert meta.get('stop_reason') in ('end_turn', 'stop_sequence'), 'Incomplete verdict'
            verdict = judge.decode_verdict(raw)
            if case['suite'] == 'advisory':
                checked = native.validate(verdict, case, answer, require_counterevidence=True)
                active = {c['id']: c for c in case['criteria'] if c['id'] in case['applicable_ids']}
                value = sum(i['level'] == native.labels(active[i['id']])[0] for i in checked['items']) / len(active)
            else:
                checked = judge.score_items(verdict, case['rubric'], 'equal')
                value = int(all(i['pass'] for i in checked['items']))
            result = {'status': 'evaluated', 'score': value, 'raw_verdict': verdict, 'metadata': meta,
                      'answer_sha256': hashlib.sha256(answer.encode()).hexdigest()}
            save(p, result); return result
        except Exception as exc:
            # Valid scores are never retried; errors are type-only to protect secrets.
            save(dest / f'judge_error_{stamp}.json', {'error_type': type(exc).__name__, 'attempt': attempt + 1})
            time.sleep(2 * (attempt + 1))
    result = {'status': 'judge_unresolved', 'score': None}
    save(p, result); return result


def summary(a, cases):
    rows = []
    for c in cases:
        for condition in a.conditions:
            p = a.out / 'cells' / condition / c['task_id'] / 'score.json'
            result = read(p) if p.exists() else {'status': 'pending', 'score': None}
            rows.append({'task_id': c['task_id'], 'suite': c['suite'], 'condition': condition, **result})
    groups = {}
    for condition in a.conditions:
        groups[condition] = {}
        for suite in ('advisory', 'core'):
            selected = [x for x in rows if x['condition'] == condition and x['suite'] == suite]
            complete = bool(selected) and all(x['score'] is not None for x in selected)
            groups[condition][suite] = {'n': len(selected), 'complete': complete,
                'mean': sum(x['score'] for x in selected) / len(selected) if complete else None}
    save(a.out / 'status.json', {'at': now(), 'counts': dict(__import__('collections').Counter(x['status'] for x in rows)),
                               'target': len(rows), 'groups': groups})
    save(a.out / 'summary.json', {'at': now(), 'groups': groups,
        'advisory_metric': 'highest-native-level dimension fraction', 'core_metric': 'whole-task all-PASS',
        'combined305_mean': None, 'candidate_scope': 'task-local candidate package', 'functional_route_accuracy': None})


async def execute(a, cases, candidates):
    from benchflow import RolloutConfig
    harness.OUT = a.out
    harness.configure_adapter(); runtime.install_runtime_fixes()
    cls = runtime.recovery_rollout_class(harness, asyncio.Semaphore(2), save)
    token = 'native-' + secrets.token_hex(16)
    bridge = Bridge(a, token); await bridge.start()
    base = f'http://{harness.gateway()}:{a.port}/v1'
    os.environ.update(BENCHFLOW_PROVIDER_BASE_URL=base, BENCHFLOW_PROVIDER_API_KEY=token,
        NO_PROXY=f'localhost,127.0.0.1,::1,{harness.gateway()}', no_proxy=f'localhost,127.0.0.1,::1,{harness.gateway()}')
    pool = ProcessPoolExecutor(max_workers=a.judge_concurrency)
    loop, sem, scoring = asyncio.get_running_loop(), asyncio.Semaphore(a.concurrency), []
    async def cell(c, condition):
        dest = a.out / 'cells' / condition / c['task_id']
        p = dest / 'input.json'
        if p.exists(): x = read(p)
        else:
            async with sem:
                obj = None
                try:
                    cfg = RolloutConfig(task_path=a.out / 'tasks' / condition / c['task_id'], agent='opencode',
                        model='vllm/' + a.model, environment='docker', skill_mode='with-skill' if condition == 'with_skill' else 'no-skill',
                        skip_verify=True, timeout=600, agent_idle_timeout=600, jobs_dir=str(a.out / 'jobs' / condition / c['task_id']),
                        agent_env={'OPENAI_API_KEY': token, 'OPENAI_BASE_URL': base,
                            'OPENCODE_CONFIG_CONTENT': json.dumps({'default_agent': 'build', 'agent': {'build': {'prompt': harness.EDUCATION_PROMPT}}})})
                    obj = await cls.create(cfg); result = await obj.run(); job = obj._rollout_dir
                    error = 'execution_failure' if result.error or getattr(result, 'error_category', None) else None
                except Exception as exc:
                    error = 'execution_' + type(exc).__name__; job = obj._rollout_dir if obj else None
                dbp = Path(job) / 'capture/opencode_database.json' if job else None
                db = read(dbp) if dbp and dbp.exists() else {}
                answer, extraction = final_answer(db)
                error = error or (None if extraction['terminal'] else 'answer_extraction_failure')
                bank = candidates.get(c['task_id'], [c['task_id'].split('__')[0]]) if condition == 'with_skill' else []
                x = {'case': c, 'answer': answer, 'extraction': extraction, 'error': error,
                     'job': str(job) if job else None, 'cell': str(dest), 'judge_base_url': a.judge_base_url}
                save(p, x); (dest / 'answer.md').write_text(answer + '\n')
                save(dest / 'routing.json', route_audit(db, c, bank))
                if error: save(dest / 'score.json', {'status': error, 'score': 0., 'score_origin': 'terminal_failure_not_judge'})
        if not x['error']: scoring.append(loop.run_in_executor(pool, score_one, x))
        summary(a, cases)
    try:
        await asyncio.gather(*(cell(c, condition) for c in cases for condition in a.conditions))
        await bridge.stop()
        await asyncio.gather(*scoring)
        summary(a, cases)
    finally:
        await bridge.stop(); pool.shutdown(wait=True, cancel_futures=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, default=Path('data/exports/eduskillbench-305-20261003/cases.json'))
    p.add_argument('--skills', type=Path, default=Path('skills/single_turn'))
    p.add_argument('--candidate-map', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--policy-base-url', required=True)
    p.add_argument('--policy-options', default='{}')
    p.add_argument('--judge-base-url', default='https://ark.cn-beijing.volces.com/api/plan/v1')
    p.add_argument('--conditions', nargs='+', choices=['no_skill', 'with_skill'], default=['no_skill', 'with_skill'])
    p.add_argument('--task-id', action='append')
    p.add_argument('--image', default='eduskillbench-opencode-runtime:1.18.11-rg1')
    p.add_argument('--port', type=int, default=8025)
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--judge-concurrency', type=int, default=64)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not 1 <= a.concurrency <= 64 or not 1 <= a.judge_concurrency <= 128: p.error('Invalid concurrency')
    if len(set(a.conditions)) != len(a.conditions): p.error('Duplicate conditions')
    if 'api' in a.policy_options.lower() or 'secret' in a.policy_options.lower(): p.error('Do not put credentials in policy options')
    a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases = read(a.cases)
        if a.task_id:
            requested = set(a.task_id); cases = [c for c in cases if c['task_id'] in requested]
            assert requested == {c['task_id'] for c in cases}, 'Unknown task ID'
        assert cases and len({c['task_id'] for c in cases}) == len(cases)
        candidates = read(a.candidate_map) if a.candidate_map else {}
        for c in cases:
            for condition in a.conditions: build_task(c, condition, a.out, a.skills, a.image, candidates)
        manifest = {'cases_sha256': sha(a.cases), 'task_ids': [c['task_id'] for c in cases],
            'model': a.model, 'policy_base_url': a.policy_base_url, 'policy_options': json.loads(a.policy_options),
            'judge_model': 'deepseek-v4-pro-ga-260813', 'judge_base_url': a.judge_base_url,
            'conditions': a.conditions, 'image_id': json.loads(subprocess.check_output(['docker', 'image', 'inspect', a.image], text=True))[0]['Id'],
            'system_prompt': harness.EDUCATION_PROMPT,
            'resources': {str(x.relative_to(a.out)): sha(x) for x in (a.out / 'tasks').rglob('*') if x.is_file()},
            'concurrency': a.concurrency, 'judge_concurrency': a.judge_concurrency,
            'agent_budget_seconds': 600, 'provider_budget_seconds': 480, 'max_tokens': 10000,
            'skill_mode': 'task-local native access; no forced Skill text; no full-library search'}
        if (a.out / 'manifest.json').exists(): assert read(a.out / 'manifest.json') == manifest, 'Frozen configuration changed'
        else: save(a.out / 'manifest.json', manifest)
        if not a.execute:
            print(json.dumps({'prepared': True, 'cells': len(cases) * len(a.conditions), 'out': str(a.out)})); return
        if not os.getenv('LLM_API_KEY'): p.error('LLM_API_KEY missing')
        asyncio.run(execute(a, cases, candidates))


if __name__ == '__main__': main()
