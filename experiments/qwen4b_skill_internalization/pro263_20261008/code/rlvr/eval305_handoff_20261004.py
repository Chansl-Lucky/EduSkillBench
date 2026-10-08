"""Qwen three-arm Docker evaluation using the published 5a9c6c2 handoff.

Restored BenchFlow sources and dependencies match the handoff. The experiment
adapter uses the existing prebaked Node/OpenCode runtime, normalizes public
Skill permissions, and preserves native OpenCode evidence before cleanup.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from dataclasses import replace
import fcntl
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

from rlvr.eval305_20261004 import ROOT, BASE, ADAPTER, JUDGE, API, ARMS, now, digest, read, rows, save, append, require, skill_id

UP=ROOT/'.local-artifacts/handoff_5a9c6c2'
OUT=ROOT/'artifacts/eval305_handoff_sft246_20261004_r2'
IMAGE='eduskillbench-opencode-runtime:1.18.11-rg1'
PORT=8015
BASE_ALIAS='qwen3-4b-eval305-base'
SFT_ALIAS='qwen3-4b-eval305-sft246'
SMOKE_IDS=['hinge-question-designer__01','lesson-builder__cn01_01']
CONCURRENCY=8
sys.path.insert(0,str(UP))
from repro import judge as native_judge, source_protocol as native
from repro.profiles import EDUCATION_PROMPT

def event(kind,**kw):
    record={'at':now(),'event':kind,**kw};append(OUT/'events.jsonl',record)
    print(json.dumps(record,ensure_ascii=False),flush=True)

def gateway():
    return subprocess.check_output(['docker','network','inspect','bridge','--format','{{range .IPAM.Config}}{{.Gateway}}{{end}}'],text=True).strip()

def task_path(arm,tid):
    mode='with_skill' if arm=='base_skill' else 'no_skill'
    return OUT/'tasks'/mode/('handoff305_'+mode+'_'+tid)

def prepare():
    if (OUT/'manifest.json').exists():
        m=read(OUT/'manifest.json')
        require(m['adapter_sha256']==digest((ADAPTER/'adapter_model.safetensors').read_bytes()),'adapter changed')
        for name,h in m['input_hashes'].items():require(digest((OUT/name).read_bytes())==h,'frozen input changed: '+name)
        require(m['runner_sha256']==digest(Path(__file__).read_bytes()),'runner changed; create a fresh run')
        return
    OUT.mkdir(parents=True,exist_ok=True)
    source=UP/'data/exports/eduskillbench-305-20261003/cases.json'
    cases=read(source);require(len(cases)==305,'expected 305 cases')
    (OUT/'cases.json').write_bytes(source.read_bytes())
    for name in ['judge.py','source_protocol.py','profiles.py','prepare.py','runner.py','source_runner.py','paired_runner.py','api_config_formal.json']:
        target=OUT/'upstream/repro'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((UP/'repro'/name).read_bytes())
    for sid in sorted({skill_id(c) for c in cases}):
        shutil.copytree(UP/'skills/single_turn'/sid,OUT/'upstream/skills'/sid,ignore=shutil.ignore_patterns('evals','__pycache__','.git'))
    for c in cases:
        for arm in ['base','base_skill']:
            task=task_path(arm,c['task_id']);env=task/'environment';env.mkdir(parents=True)
            prompt=c['context']+'\n\n'+c['user_prompt']+'\n'
            docker='FROM '+IMAGE+'\nRUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app /tests\n'
            skill_setting=''
            if arm=='base_skill':
                sid=skill_id(c);src=OUT/'upstream/skills'/sid
                prompt+='\n## Required procedure\n'+(src/'SKILL.md').read_text()
                dest=env/'skills'/sid;shutil.copytree(src,dest)
                # File modes are runtime input too. Preserve executable resources;
                # all public files must be readable by the sandbox's agent user.
                for p in [env/'skills',*sorted((env/'skills').rglob('*'))]:
                    p.chmod(0o755 if p.is_dir() or p.stat().st_mode&0o111 else 0o644)
                docker+='COPY skills/ /skills/\nRUN find /skills -type d -exec chmod 755 {} + && find /skills -type f -exec chmod a+r {} +\n'
                skill_setting='\nskills_dir = "/skills"'
            (task/'instruction.md').write_text(prompt)
            (env/'Dockerfile').write_text(docker+'WORKDIR /app\n')
            (task/'task.toml').write_text('version = "1.0"\n[metadata]\nauthor_name="EduSkillBench"\ndifficulty="medium"\ncategory="education"\n[agent]\ntimeout_sec=600\n[verifier]\ntimeout_sec=1900\n[environment]\ncpus=1\nmemory_mb=1024\nallow_internet=true'+skill_setting+'\n')
    hashes={p.relative_to(OUT).as_posix():digest(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}
    image=read_image()
    save(OUT/'manifest.json',dict(at=now(),source_commit='5a9c6c2b4f4792e4539f8c4da801f9c1f789fc7f',
        cases=305,cells=915,input_hashes=hashes,runner_sha256=digest(Path(__file__).read_bytes()),
        adapter=str(ADAPTER),adapter_sha256=digest((ADAPTER/'adapter_model.safetensors').read_bytes()),
        system_profile='education-single-turn',skill_mode='forced SKILL.md plus complete accessible resources',
        core_metric='weighted primary, equal secondary',advisory_metric='source native levels; no invented point score',
        judge_model=JUDGE,judge_api=API,judge_max_tokens=10000,agent_timeout=600,
        environment_concurrency=CONCURRENCY,judge_concurrency=64,image_id=image['Id'],image_reference=IMAGE,
        framework_identity=read(ROOT/'artifacts/handoff_audit_20261004/restored_identity.json'),
        explicit_adaptations=['Local vLLM Qwen base/SFT instead of remote policies',
            'DSV4 Flash 0731 instead of handoff DS Pro judge, per user requirement',
            'Prebaked Node22.20/OpenCode1.18.11 avoids upstream machine-specific cache8123',
            'Public Skill permissions normalized and verified as agent',
            'Read-only wire/database capture supplements framework evidence before teardown'],
        handoff_chat_is_a_separate_protocol=True))

def read_image():return json.loads(subprocess.check_output(['docker','image','inspect',IMAGE],text=True))[0]

def configure_adapter():
    from benchflow.agents.registry import AGENTS, AGENT_INSTALLERS, _opencode_family_proxy_wrapper_install
    cfg=AGENTS['opencode']
    # The immutable local image already contains these exact runtime versions.
    # Keep the upstream discovery path and provider wrapper unchanged.
    install=('test -x /opt/benchflow/node/bin/node && test -x /opt/benchflow/js-agents/bin/opencode && '
        '/opt/benchflow/node/bin/node --version && /opt/benchflow/js-agents/bin/opencode --version && '+
        _opencode_family_proxy_wrapper_install('opencode','.config/opencode/opencode.json'))
    AGENTS['opencode']=replace(cfg,install_cmd=install)
    AGENT_INSTALLERS['opencode']=install
    from benchflow.acp.container_transport import ContainerTransport
    original=ContainerTransport.receive
    async def receive(self):
        message=await original(self)
        if self._agent_log_path:
            dest=self._agent_log_path.parent.parent/'trajectory/acp_wire.jsonl'
            append(dest,message)
        return message
    ContainerTransport.receive=receive

DB_EXPORT="""import sqlite3,json,pathlib
p=pathlib.Path('/home/agent/.local/share/opencode/opencode.db')
dest=pathlib.Path('/logs/agent/opencode_database.json')
if p.exists():
 con=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
 tables={r[0] for r in con.execute('select name from sqlite_master where type=\"table\"')}
 data={name:[dict(r) for r in con.execute('select * from '+name)] for name in ['session','message','part'] if name in tables}
 dest.write_text(json.dumps(data,ensure_ascii=False));con.close()
"""

def audited_rollout_class():
    from benchflow.rollout import Rollout
    class AuditedRollout(Rollout):
        async def install_agent(self):
            await super().install_agent()
            audit=self._require_rollout_dir()/'capture';audit.mkdir(exist_ok=True)
            probe=await self._env.exec("runuser -u agent -- sh -c 'if [ -d /skills ]; then find -L /home/agent/.config/opencode/skills -name SKILL.md -type f -exec test -r {} \\; -print; else echo NO_SKILL_DIRECTORY; fi'",timeout_sec=15)
            save(audit/'skill_readability.json',dict(return_code=probe.return_code,stdout=probe.stdout,stderr=probe.stderr))
            if self._config.skill_mode=='with-skill':
                require(probe.return_code==0 and 'SKILL.md' in (probe.stdout or ''),'agent cannot read required Skill')
        async def cleanup(self):
            if self._env:
                dest=self._require_rollout_dir()/'capture';dest.mkdir(exist_ok=True)
                # SQLite and logs are collected while the container still exists,
                # including failed and timed-out rollouts. Never include auth files.
                import shlex
                try:
                    result=await self._env.exec('python3 -c '+shlex.quote(DB_EXPORT),timeout_sec=15)
                    save(dest/'db_export_status.json',dict(return_code=result.return_code,stderr=result.stderr))
                    await self._env.download_file('/logs/agent/opencode_database.json',dest/'opencode_database.json')
                except Exception as e:save(dest/'db_export_error.json',dict(error=str(e)[:300]))
                try:await self._env.download_dir('/home/agent/.local/share/opencode/log',dest/'opencode_logs')
                except Exception as e:save(dest/'log_export_error.json',dict(error=str(e)[:300]))
                try:
                    result=await self._env.exec("python3 -c 'import json,pathlib; p=pathlib.Path(\"/home/agent/.config/opencode/opencode.json\"); d=json.loads(p.read_text()); [(v.get(\"options\",{}).pop(\"apiKey\",None)) for v in d.get(\"provider\",{}).values()]; print(json.dumps(d))'",timeout_sec=10)
                    if result.return_code==0:save(dest/'opencode_config.redacted.json',json.loads(result.stdout))
                except Exception as e:save(dest/'config_capture_error.json',dict(error=str(e)[:300]))
            await super().cleanup()
    return AuditedRollout

def captured_evidence(job):
    db=read(job/'capture/opencode_database.json') if (job/'capture/opencode_database.json').exists() else {}
    parts=[];answer_chars=0;tool_count=0;messages={}
    for row in db.get('message',[]):
        data=json.loads(row['data']) if isinstance(row.get('data'),str) else row.get('data',{})
        messages[row['id']]={**data,'session_id':row.get('session_id')}
    for row in db.get('part',[]):
        data=json.loads(row['data']) if isinstance(row.get('data'),str) else row.get('data',{})
        msg=messages.get(row.get('message_id'),{})
        if msg.get('role')!='assistant' or data.get('type') not in ['text','tool']:continue
        if data.get('type')=='text':answer_chars+=len(data.get('text') or '')
        else:tool_count+=1
        parts.append(dict(session_id=msg.get('session_id'),message_id=row.get('message_id'),**data))
    acp=[]
    wire=[]
    for row in rows(job/'trajectory/acp_wire.jsonl'):
        if row.get('method')=='session/update':
            params=row.get('params',{});update=params.get('update',{})
            if update.get('sessionUpdate') in ['agent_message_chunk','tool_call','tool_call_update']:wire.append(params)
    for row in rows(job/'trajectory/acp_trajectory.jsonl'):
        if row.get('type') in ['agent_message','tool_call']:acp.append(row)
    # Database session IDs keep sub-agent content distinct from parent deliverables.
    evidence={'opencode_sessions':db.get('session',[]),'opencode_assistant_parts':parts,'acp_events':acp,'wire_updates':wire}
    text=json.dumps(evidence,ensure_ascii=False,indent=2)
    if len(text)>200000:text=text[:120000]+'\n[... middle omitted ...]\n'+text[-80000:]
    return text,dict(assistant_text_chars=answer_chars,database_tool_parts=tool_count,wire_updates=len(wire),
        has_deliverable=answer_chars>0 or any(p.get('type')=='tool' and p.get('state',{}).get('status')=='completed' for p in parts),
        database_captured=bool(db))

def report(phase,**extra):
    rr={(r['arm'],r['task_id']):r for r in rows(OUT/'rollouts.jsonl')}
    ss={(r['arm'],r['task_id']):r for r in rows(OUT/'scores.jsonl') if r['status']=='ok'}
    summary={}
    for arm in ARMS:
        valid=[s for k,s in ss.items() if k[0]==arm]
        core=[s['score'] for s in valid if s['suite']=='core']
        summary[arm]=dict(attempted=sum(k[0]==arm for k in rr),judged=len(valid),
            failed=sum(bool(r.get('error')) for k,r in rr.items() if k[0]==arm),core_n=len(core),
            core_weighted_mean=statistics.mean(core) if core else None,
            advisory_n=sum(s['suite']=='advisory' for s in valid))
    state=dict(at=now(),phase=phase,target=915,rollouts=len(rr),judged=len(ss),by_arm=summary,**extra)
    save(OUT/'status.json',state);save(OUT/'comparison.json',summary)
    return state

async def score(c,r,judge,limit):
    from rlvr.judge import _extract_json
    if r.get('error') or not r['evidence']['has_deliverable']:return
    text,ev=captured_evidence(Path(r['job']))
    folder=OUT/'judge'/r['arm']/c['task_id'];folder.mkdir(parents=True,exist_ok=True)
    if c['suite']=='core':
        case={'question':c['context']+'\n\n'+c['user_prompt'],'ground_truth':c['expected_output'],
            'rubric':[{'id':f'C{i+1}',**v} for i,v in enumerate(c['rubric'])],'score_metric':'weighted'}
        prompt=native_judge.build_prompt(case,text)
        validate=lambda p:native_judge.score_items(p,case['rubric'],'weighted')
    else:
        prompt=native.prompt(c,text)
        validate=lambda p:native.validate(p,c,text,require_counterevidence=True)
    def record_verdict(body,verdict):
        record={**verdict,'at':now(),'arm':r['arm'],'task_id':c['task_id'],'suite':c['suite'],
            'skill_id':skill_id(c),'status':'ok','judge_model':body['model'],
            'trajectory_sha256':digest(text),'usage':body.get('usage')}
        append(OUT/'scores.jsonl',record)
        event('scored',arm=r['arm'],task_id=c['task_id'],score=record.get('score'))
    # A valid stored response survives a ledger/export bug or API outage. Reuse
    # it only after checking this exact prompt/evidence and native validation.
    for response in sorted(folder.glob('response_*.json')):
        try:
            request=read(folder/'request_0.json')
            require(request['messages'][0]['content']==prompt,'cached judge prompt changed')
            body=read(response);require(body.get('model')==JUDGE,'cached judge model changed')
            ch=body['choices'][0];require(ch.get('finish_reason')=='stop','cached judge incomplete')
            verdict=validate(_extract_json(ch['message'].get('content') or ''))
            record_verdict(body,verdict);return
        except (ValueError,KeyError,TypeError):continue
    save(folder/'evidence.json',dict(text=text,**ev))
    payload={'model':JUDGE,'messages':[{'role':'user','content':prompt}],'max_tokens':10000,'temperature':0,'stream':False,'thinking':{'type':'disabled'}}
    async with limit:
        for attempt in range(2):
            save(folder/f'request_{attempt}.json',payload)
            try:
                body=await judge._post_with_retries(payload);save(folder/f'response_{time.time_ns()}.json',body)
                require(body.get('model')==JUDGE,'judge model version mismatch')
                ch=body['choices'][0];require(ch.get('finish_reason')=='stop','judge response incomplete')
                parsed=_extract_json(ch['message'].get('content') or '');verdict=validate(parsed)
                record_verdict(body,verdict);return
            except Exception as e:
                err=re.sub(r'(?:ark-|sk-)[A-Za-z0-9-]+','[REDACTED]',str(e))[:500]
                append(folder/'errors.jsonl',dict(at=now(),attempt=attempt,error=err))
                payload['messages']=[{'role':'user','content':prompt+'\nPrevious verdict failed validation: '+err+'. Return complete corrected JSON; do not change the rubric.'}]
        append(OUT/'scores.jsonl',dict(arm=r['arm'],task_id=c['task_id'],status='error',error=err))
        event('judge_failed',arm=r['arm'],task_id=c['task_id'],error=err)

async def rollout(c,arm,limit):
    from benchflow import RolloutConfig
    cls=audited_rollout_class();tid=c['task_id']
    async with limit:
        event('rollout_start',arm=arm,task_id=tid)
        cfg=RolloutConfig(task_path=task_path(arm,tid),agent='opencode',model='vllm/'+(SFT_ALIAS if arm=='sft' else BASE_ALIAS),
            environment='docker',skill_mode='with-skill' if arm=='base_skill' else 'no-skill',skip_verify=True,
            timeout=600,agent_idle_timeout=600,jobs_dir=str(OUT/'jobs'/arm/tid),
            agent_env={'OPENAI_API_KEY':'local-eval305','OPENAI_BASE_URL':f'http://{gateway()}:{PORT}/v1',
                'OPENCODE_CONFIG_CONTENT':json.dumps({'default_agent':'build','agent':{'build':{'prompt':EDUCATION_PROMPT}}})})
        start=time.monotonic();obj=await cls.create(cfg);result=await obj.run()
        paths=list((OUT/'jobs'/arm/tid).rglob('result.json'));require(bool(paths),'result export missing')
        job=max(paths,key=lambda p:p.stat().st_mtime).parent
        text,ev=captured_evidence(job)
        save(job/'capture/evidence_manifest.json',ev)
        rec=dict(at=now(),arm=arm,task_id=tid,job=str(job),suite=c['suite'],
            error=result.error,error_category=getattr(result,'error_category',None),
            elapsed_seconds=round(time.monotonic()-start,2),tool_calls=result.n_tool_calls,evidence=ev)
        append(OUT/'rollouts.jsonl',rec);event('rollout_done',**rec);return rec

async def pipeline(smoke_only=False):
    import httpx
    from rlvr.judge import TokenPlanJudge
    prepare();configure_adapter()
    base=f'http://{gateway()}:{PORT}/v1'
    os.environ.update(BENCHFLOW_PROVIDER_BASE_URL=base,BENCHFLOW_PROVIDER_API_KEY='local-eval305',
        OPENAI_BASE_URL=base,OPENAI_API_KEY='local-eval305',NO_PROXY=f'localhost,127.0.0.1,::1,{gateway()}',
        no_proxy=f'localhost,127.0.0.1,::1,{gateway()}',DOCKER_CONFIG=str(ROOT/'config/docker-proxy'))
    args=read(ROOT/'artifacts/eval305_harness_sft246_20261004_v2/server_command.json')['argv']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'};env.pop('ARK_API_KEY',None)
    server=None;judge=TokenPlanJudge(os.environ['ARK_API_KEY'],JUDGE,API,concurrency=64,timeout=120)
    try:
        check=await judge._post_with_retries(dict(model=JUDGE,messages=[{'role':'user','content':'Reply OK.'}],max_tokens=16,stream=False,thinking={'type':'disabled'}))
        require(check.get('model')==JUDGE,'judge preflight model mismatch');save(OUT/'judge_preflight.json',{'at':now(),'model':JUDGE,'ok':True})
        with (OUT/'server.log').open('ab') as log:
            server=await asyncio.create_subprocess_exec(*args,cwd=ROOT,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
        save(OUT/'server_pid.json',dict(pid=server.pid,at=now()));report('loading_model',server_pid=server.pid)
        async with httpx.AsyncClient(trust_env=False,timeout=15,headers={'Authorization':'Bearer local-eval305'}) as client:
            for _ in range(120):
                require(server.returncode is None,'server exited')
                try:
                    resp=await client.get(base+'/models');resp.raise_for_status()
                    require({BASE_ALIAS,SFT_ALIAS}<={v['id'] for v in resp.json()['data']},'model aliases missing');break
                except (httpx.HTTPError,ValueError):await asyncio.sleep(5)
            else:raise RuntimeError('model readiness timeout')
        cases=read(OUT/'cases.json');lookup={c['task_id']:c for c in cases}
        agents=asyncio.Semaphore(CONCURRENCY);judges=asyncio.Semaphore(64);pending=[]
        async def cell(c,a):
            cache={(r['arm'],r['task_id']):r for r in rows(OUT/'rollouts.jsonl')}
            r=cache.get((a,c['task_id']))
            if r is None:r=await rollout(c,a,agents)
            done={(r['arm'],r['task_id']) for r in rows(OUT/'scores.jsonl') if r['status']=='ok'}
            if (a,c['task_id']) not in done:pending.append(asyncio.create_task(score(c,r,judge,judges)))
            return r
        report('smoke_running',server_pid=server.pid)
        smoke_tasks=[asyncio.create_task(cell(lookup[t],a)) for t in SMOKE_IDS for a in ARMS]
        while any(not t.done() for t in smoke_tasks):
            report('smoke_running',server_pid=server.pid);await asyncio.sleep(15)
        smokes=await asyncio.gather(*smoke_tasks);await asyncio.gather(*pending);pending.clear()
        done={(r['arm'],r['task_id']) for r in rows(OUT/'scores.jsonl') if r['status']=='ok'}
        passed=all(not r.get('error') and r['evidence']['has_deliverable'] and r['evidence']['database_captured'] for r in smokes) and all((a,t) in done for t in SMOKE_IDS for a in ARMS)
        save(OUT/'SMOKE_GATE.json',dict(at=now(),passed=passed,checks=smokes,scored=len(done)))
        require(passed,'smoke gate failed; inspect captured OpenCode logs; full run not launched')
        event('smoke_passed',cells=6)
        if smoke_only:report('smoke_complete',gpu_models_unloaded=False);return
        tasks=[asyncio.create_task(cell(c,a)) for c in cases for a in ARMS if c['task_id'] not in SMOKE_IDS]
        while any(not t.done() for t in tasks):
            report('formal_running',server_pid=server.pid);await asyncio.sleep(30)
        await asyncio.gather(*tasks)
        os.killpg(server.pid,signal.SIGTERM);await server.wait()
        report('judging',gpu_models_unloaded=True);await asyncio.gather(*pending)
        # Retry scoring only; never repeat a completed policy rollout for judge failure.
        done={(r['arm'],r['task_id']) for r in rows(OUT/'scores.jsonl') if r['status']=='ok'}
        latest={(r['arm'],r['task_id']):r for r in rows(OUT/'rollouts.jsonl')}
        await asyncio.gather(*(score(lookup[t],r,judge,judges) for (a,t),r in latest.items() if (a,t) not in done))
        state=report('finished',gpu_models_unloaded=True);save(OUT/'FINISHED.json',state)
    finally:
        if server is not None and server.returncode is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:await asyncio.wait_for(server.wait(),30)
            except asyncio.TimeoutError:os.killpg(server.pid,signal.SIGKILL);await server.wait()
        await judge.aclose()

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','run','status']);p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    if a.mode=='status':print(json.dumps(read(OUT/'status.json'),ensure_ascii=False,indent=2));return
    prepare()
    if a.mode=='prepare':print('Prepared 305 cases and 610 tasks from handoff.');return
    require(bool(os.environ.get('ARK_API_KEY')),'ARK_API_KEY missing')
    if a.mode=='launch':
        lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        argv=[sys.executable,'-u','-m','rlvr.eval305_handoff_20261004','run']+(['--smoke-only'] if a.smoke_only else [])
        with (OUT/'worker.log').open('ab') as log:
            proc=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'code'),'PIPELINE_LOCK_FD':str(lock.fileno())},
                pass_fds=(lock.fileno(),),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(OUT/'launcher.json',dict(pid=proc.pid,at=now(),smoke_only=a.smoke_only));print(json.dumps(read(OUT/'launcher.json')));return
    try:asyncio.run(pipeline(a.smoke_only))
    except Exception as e:
        report('blocked_smoke_or_runtime',error=re.sub(r'(?:ark-|sk-)[A-Za-z0-9-]+','[REDACTED]',str(e))[:500],gpu_models_unloaded=True);raise

if __name__=='__main__':
    logging.basicConfig(level=logging.WARNING)
    main()
