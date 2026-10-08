#!/usr/bin/env python3
"""Evaluate the completed SFT246->GRPO adapter on advisory263, no Skill.

Reuse the previous Docker/OpenCode generation protocol and pinned Pro scorer.
Only GRPO is generated. GPU server always exits; scoring caches survive exit.
"""
from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'code/evaluation'))
from rlvr import eval305_handoff_20261004 as harness
import rescore_advisory263_pro_20261007 as scorer

OUT=ROOT/'artifacts/eval_grpo263_pro_20261008'
OLD=harness.OUT
PREVIOUS_PRO=scorer.OUT
ADAPTER=ROOT/'artifacts/sft246_grpo_20261005_r2/final_adapter'
ALIAS='qwen3-4b-advisory263-sft246-grpo123'
PORT=8018
ENV_CONCURRENCY=8
JUDGE_CONCURRENCY=64

def save(p,x):scorer.save(p,x)
def read(p):return scorer.read(p)
def rows(p):return scorer.rows(p) if Path(p).exists() else []
def now():return scorer.now()
def sha(p):return scorer.sha(p)

def event(name,**kw):
    r={'at':now(),'event':name,**kw};scorer.append(OUT/'events.jsonl',r)
    print(json.dumps(r,ensure_ascii=False),flush=True)

def setup_worker():
    scorer.OUT=OUT
    os.environ['ANTHROPIC_BASE_URL']='https://ark.cn-beijing.volces.com/api/plan/v1'
    os.environ.pop('ANTHROPIC_API_KEY',None);os.environ.pop('ANTHROPIC_AUTH_TOKEN',None)
    proxy=os.environ.get('ARK_SCORING_PROXY')
    if proxy:
        # Only the network route changes. Preserve pinned prompt, schema,
        # request payload, streaming parser and all upstream deadlines.
        scorer.judge.request_transport=lambda url:(
            urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy})).open,
            'explicit_ark_proxy')

def grade(x):return scorer.score_one(x)

def configure_harness():
    harness.OUT=OUT;harness.PORT=PORT;harness.SFT_ALIAS=ALIAS
    # Old rollout selects SFT alias for arm='sft'; replace base alias as well so
    # arm='grpo' unambiguously requests the adapter, never the untouched base.
    harness.BASE_ALIAS=ALIAS
    harness.configure_adapter()

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    complete=read(ROOT/'artifacts/sft246_grpo_20261005_r2/FINISHED.json')
    assert complete['global_step']==123 and complete['adapter_sha256']==sha(ADAPTER/'adapter_model.safetensors')
    cases=[c for c in read(OLD/'cases.json') if c['suite']=='advisory'];assert len(cases)==263
    upstream={c['task_id']:c for c in read(scorer.UP/'data/exports/eduskillbench-305-20261003/cases.json') if c['suite']=='advisory'}
    assert {c['task_id']:c for c in cases}==upstream
    source_sha=sha(Path(__file__))
    if (OUT/'manifest.json').exists():
        m=read(OUT/'manifest.json');assert m['runner_sha256']==source_sha and m['adapter_sha256']==complete['adapter_sha256']
        return cases
    save(OUT/'cases.json',cases)
    task_hashes={}
    for c in cases:
        tid=c['task_id'];name='handoff305_no_skill_'+tid
        src=OLD/'tasks/no_skill'/name;dest=OUT/'tasks/no_skill'/name
        shutil.copytree(src,dest,dirs_exist_ok=True)
        for rel in ('instruction.md','task.toml','environment/Dockerfile'):
            assert sha(src/rel)==sha(dest/rel)
            task_hashes[tid+'/'+rel]=sha(dest/rel)
    argv=read(ROOT/'artifacts/eval305_harness_sft246_20261004_v2/server_command.json')['argv']
    argv=list(argv);argv[argv.index('--port')+1]=str(PORT)
    argv[argv.index('--lora-modules')+1]=ALIAS+'='+str(ADAPTER)
    save(OUT/'server_command.json',{'argv':argv,'gpu':1,'adaptations':['new port','new LoRA path/alias only']})
    # No changes to dtype, context, tool parser, sequence/batched-token limits,
    # eager mode, GPU memory ratio or task timeout.
    save(OUT/'manifest.json',{'at':now(),'tasks':263,'arm':'grpo','condition':'no_skill',
        'adapter':str(ADAPTER),'adapter_sha256':complete['adapter_sha256'],
        'runner_sha256':source_sha,'scoring_runner_sha256':sha(Path(scorer.__file__)),
        'cases_sha256':sha(OUT/'cases.json'),'task_hashes':task_hashes,
        'generation_reference':str(OLD),'system_profile':harness.EDUCATION_PROMPT,
        'environment_concurrency':ENV_CONCURRENCY,'judge_concurrency':JUDGE_CONCURRENCY,
        'policy_timeout_seconds':600,'image':harness.IMAGE,
        'server_command':argv,'scoring_profile':read(scorer.PROFILE),
        'generation_environment_difference':'Docker/OpenCode; upstream formal generation is direct Chat',
        'smoke_ids':['lesson-builder__cn01_01','hinge-question-designer__cn29_01'],
        'publish_target':'https://github.com/Chansl-Lucky/EduSkillBench',
        'no_base_or_sft_regeneration':True})
    return cases

def progress(cases,phase,**kw):
    rr={x['task_id']:x for x in rows(OUT/'rollouts.jsonl')}
    counts={};good=[]
    for c in cases:
        p=OUT/'cells/grpo'/c['task_id']
        if (p/'result.json').exists():
            x=read(p/'result.json');counts[x['status']]=counts.get(x['status'],0)+1
            if x['status']=='evaluated':good.append(x['report_score'])
        elif (p/'error.json').exists():counts['judge_unresolved']=counts.get('judge_unresolved',0)+1
    state={'at':now(),'phase':phase,'target':263,'rollouts':len(rr),'scoring':counts,
           'valid_score_mean_secondary':sum(good)/len(good) if good else None,**kw}
    save(OUT/'status.json',state);return state

def frozen_input(case,record):
    folder=OUT/'cells/grpo'/case['task_id'];db_path=Path(record['job'])/'capture/opencode_database.json'
    if not db_path.exists():db_path=Path(record['job'])/'agent/opencode_database.json'
    answer,extraction=scorer.extract_final(read(db_path)) if db_path.exists() else ('',{'terminal':False,'reason':'missing_database'})
    category='execution_failure' if record.get('error') else ('answer_extraction_failure' if not extraction['terminal'] else None)
    x={'arm':'grpo','task_id':case['task_id'],'case':case,'answer':answer,'job':record['job'],
       'source_error':record.get('error'),'category':category,'extraction':extraction,
       'source_database':str(db_path),'source_database_sha256':sha(db_path) if db_path.exists() else None}
    x['binding']=scorer.digest({'case':case,'answer':answer,'profile':read(scorer.PROFILE),'extraction':extraction})
    if (folder/'input.json').exists():assert read(folder/'input.json')['binding']==x['binding']
    else:save(folder/'input.json',x);(folder/'answer.md').write_text(answer+'\n')
    if category:
        save(folder/'result.json',{'arm':'grpo','task_id':case['task_id'],'binding':x['binding'],
            'status':category,'source_error':record.get('error'),'extraction':extraction,
            'report_score':0.,'failure_zero':True,'score_origin':'failure_policy_not_judge'})
    return x

async def unload(server):
    if server is not None and server.returncode is None:
        os.killpg(server.pid,signal.SIGTERM)
        try:await asyncio.wait_for(server.wait(),30)
        except asyncio.TimeoutError:os.killpg(server.pid,signal.SIGKILL);await server.wait()

def reports(cases):
    """Compose four-arm report without modifying previous three-arm scores."""
    combined=OUT/'comparison';combined.mkdir(exist_ok=True)
    jobs=[]
    for old in read(PREVIOUS_PRO/'inputs.json'):
        folder=combined/'cells'/old['arm']/old['task_id'];folder.mkdir(parents=True,exist_ok=True)
        source=PREVIOUS_PRO/'cells'/old['arm']/old['task_id']
        for name in ('result.json','error.json'):
            if (source/name).exists():shutil.copy2(source/name,folder/name)
        jobs.append(old)
    for c in cases:
        source=OUT/'cells/grpo'/c['task_id'];folder=combined/'cells/grpo'/c['task_id'];folder.mkdir(parents=True,exist_ok=True)
        if not (source/'input.json').exists():raise ValueError('missing GRPO case input')
        for name in ('result.json','error.json'):
            if (source/name).exists():shutil.copy2(source/name,folder/name)
        jobs.append(read(source/'input.json'))
    original_out,original_arms=scorer.OUT,scorer.ARMS
    scorer.OUT=combined;scorer.ARMS=('base','base_skill','sft','grpo')
    try:scorer.paper_report(jobs)
    finally:scorer.OUT=original_out;scorer.ARMS=original_arms
    report=read(combined/'paper_report.json')
    # Extra GRPO pairs follow the same paired source-cluster bootstrap formula.
    import numpy as np
    records=read(combined/'paper_rows.json');ids=sorted(c['task_id'] for c in cases)
    lookup={(r['arm'],r['task_id']):r for r in records};groups=sorted({r['source_group'] for r in records})
    ixgroups=[np.array([i for i,t in enumerate(ids) if lookup['grpo',t]['source_group']==g]) for g in groups]
    rng=np.random.default_rng(20261005)
    for baseline in ('base','base_skill','sft'):
        b=[lookup[baseline,t] for t in ids];g=[lookup['grpo',t] for t in ids]
        bv=np.array([r['score'] for r in b]);gv=np.array([r['score'] for r in g]);delta=gv-bv
        boot=[]
        for _ in range(5000):
            ix=np.concatenate([ixgroups[i] for i in rng.integers(0,len(groups),len(groups))]);boot.append(float(np.mean(delta[ix])))
        paired=[i for i in range(263) if not b[i]['failure_zero'] and not g[i]['failure_zero']]
        report['pairs'][baseline+'__grpo']={'n':263,'baseline':float(np.mean(bv)),'comparison':float(np.mean(gv)),
            'delta':float(np.mean(delta)),'ci':list(map(float,np.quantile(boot,[.025,.975]))),
            'complete_pair_n':len(paired),'complete_pair_delta':float(np.mean(delta[paired])) if paired else None,
            'cluster_count':len(groups),'wins_ties_losses':[int(np.sum(delta>1e-9)),int(np.sum(np.abs(delta)<=1e-9)),int(np.sum(delta< -1e-9))]}
    save(combined/'paper_report.json',report)
    return report

async def pipeline(cases):
    import httpx
    configure_harness();host=harness.gateway();base=f'http://{host}:{PORT}/v1'
    os.environ.update(BENCHFLOW_PROVIDER_BASE_URL=base,BENCHFLOW_PROVIDER_API_KEY='local-eval305',
        OPENAI_BASE_URL=base,OPENAI_API_KEY='local-eval305',NO_PROXY=f'localhost,127.0.0.1,::1,{host}',
        no_proxy=f'localhost,127.0.0.1,::1,{host}',DOCKER_CONFIG=str(ROOT/'config/docker-proxy'))
    profile=read(scorer.PROFILE)
    progress(cases,'judge_preflight',gpu_models_unloaded=True)
    event('judge_preflight_start',transport='explicit_proxy' if os.environ.get('ARK_SCORING_PROXY') else 'direct')
    async with httpx.AsyncClient(trust_env=False,proxy=os.environ.get('ARK_SCORING_PROXY'),timeout=60) as client:
        response=await client.post(profile['endpoint'],headers={'Authorization':'Bearer '+os.environ['LLM_API_KEY']},
            json={'model':profile['judge_model'],'messages':[{'role':'user','content':'Reply OK.'}],
                  'max_tokens':64,'thinking':{'type':'disabled'},'stream':False})
        response.raise_for_status();body=response.json();assert body['model']==profile['availability_probe_returned_model']
        save(OUT/'judge_preflight.json',{'at':now(),'requested':profile['judge_model'],'returned':body['model'],'http_status':200,
                                       'transport':'explicit_proxy' if os.environ.get('ARK_SCORING_PROXY') else 'direct'})
    server=None;pool=None;pending=[];inputs={}
    cache={x['task_id']:x for x in rows(OUT/'rollouts.jsonl')}
    targets=[c for c in cases if c['task_id'] not in cache]
    try:
        if targets:
            argv=read(OUT/'server_command.json')['argv'];env=dict(os.environ,CUDA_VISIBLE_DEVICES='1')
            for name in ('LLM_API_KEY','ARK_API_KEY','ANTHROPIC_API_KEY','ANTHROPIC_AUTH_TOKEN',
                         'PUBLISH_SUDO_PASSWORD','GH_TOKEN','GITHUB_TOKEN'):env.pop(name,None)
            with (OUT/'server.log').open('ab') as log:
                server=await asyncio.create_subprocess_exec(*argv,cwd=ROOT,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
            save(OUT/'server_pid.json',{'pid':server.pid,'at':now()});progress(cases,'loading_model',server_pid=server.pid)
            async with httpx.AsyncClient(trust_env=False,timeout=15,headers={'Authorization':'Bearer local-eval305'}) as client:
                for _ in range(120):
                    if server.returncode is not None:raise RuntimeError('vLLM exited before readiness')
                    try:
                        res=await client.get(base+'/models');res.raise_for_status()
                        assert ALIAS in {x['id'] for x in res.json()['data']};break
                    except (httpx.HTTPError,AssertionError):await asyncio.sleep(5)
                else:raise RuntimeError('vLLM readiness timeout')
            event('model_ready',adapter_sha256=sha(ADAPTER/'adapter_model.safetensors'),alias=ALIAS)
        pool=ProcessPoolExecutor(max_workers=JUDGE_CONCURRENCY,initializer=setup_worker)
        loop=asyncio.get_running_loop();sem=asyncio.Semaphore(ENV_CONCURRENCY)
        async def cell(c):
            tid=c['task_id'];r=cache.get(tid)
            if r is None:
                try:r=await harness.rollout(c,'grpo',sem)
                except Exception as exc:
                    r={'at':now(),'arm':'grpo','task_id':tid,'suite':'advisory',
                       'job':str(OUT/'jobs/grpo'/tid),'error':type(exc).__name__+': '+str(exc)[:500],
                       'error_category':'environment_or_export_exception'}
                    scorer.append(OUT/'rollouts.jsonl',r);event('rollout_exception',task_id=tid,error=r['error'])
                cache[tid]=r
            x=frozen_input(c,r);inputs[tid]=x
            if not x['category']:
                pending.append(loop.run_in_executor(pool,grade,x))
            progress(cases,'running',gpu_models_unloaded=False)
            return r
        lookup={c['task_id']:c for c in cases};smoke_ids=read(OUT/'manifest.json')['smoke_ids']
        smoke=await asyncio.gather(*(cell(lookup[t]) for t in smoke_ids))
        smokes_ok=all(not r.get('error') and inputs[r['task_id']]['extraction']['terminal'] for r in smoke)
        # Readiness gate is about generation evidence, not how high the score is.
        await asyncio.gather(*pending);pending.clear()
        smoke_scores=all((OUT/'cells/grpo'/t/'result.json').exists() for t in smoke_ids)
        save(OUT/'SMOKE_GATE.json',{'at':now(),'generation_passed':smokes_ok,'scoring_protocol_passed':smoke_scores})
        if not smokes_ok or not smoke_scores:raise RuntimeError('smoke gate failed; full generation not launched')
        event('smoke_passed',cases=2)
        await asyncio.gather(*(cell(c) for c in cases if c['task_id'] not in smoke_ids))
        await unload(server);event('gpu_unloaded')
        progress(cases,'judging',gpu_models_unloaded=True)
        await asyncio.gather(*pending);pending.clear()
        # One bounded failed-score recovery sweep, no policy regeneration.
        retry=[x for x in inputs.values() if not x['category'] and not (OUT/'cells/grpo'/x['task_id']/'result.json').exists()]
        if retry:event('judge_only_recovery',targets=len(retry))
        await asyncio.gather(*(loop.run_in_executor(pool,grade,x) for x in retry))
        save(OUT/'inputs.json',list(inputs.values()));report=reports(cases)
        state=progress(cases,'finished',gpu_models_unloaded=True,report=str(OUT/'comparison/REPORT.md'))
        save(OUT/'FINISHED.json',state);event('finished',grpo_score=report['arms']['grpo']['mean'])
        publisher=ROOT/'code/evaluation/publish_grpo263_20261008.py'
        command=[str(ROOT/'.venv/bin/python'),str(publisher),'--output',str(OUT)]
        result=await asyncio.create_subprocess_exec(*command,cwd=ROOT,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
        output,_=await result.communicate()
        (OUT/'publish.log').write_bytes(output);save(OUT/'publish_status.json',{'at':now(),'exit_code':result.returncode,'log':str(OUT/'publish.log')})
        event('publish_complete' if result.returncode==0 else 'publish_failed',exit_code=result.returncode)
    finally:
        await unload(server)
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'pipeline.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=prepare()
        if a.prepare_only:print(json.dumps({'prepared':263,'adapter':str(ADAPTER)}));return
        if not os.environ.get('LLM_API_KEY'):raise RuntimeError('LLM_API_KEY missing')
        asyncio.run(pipeline(cases))

if __name__=='__main__':
    try:main()
    except BlockingIOError:
        raise SystemExit('This output directory already has an owner; no status overwritten.')
    except Exception as exc:
        save(OUT/'status.json',{'at':now(),'phase':'failed','error':str(exc)[:1000],'type':type(exc).__name__})
        event('failed',error=str(exc)[:1000]);raise
