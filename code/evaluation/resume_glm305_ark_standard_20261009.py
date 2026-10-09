#!/usr/bin/env python3
"""Transport-only resume of the frozen r4 judge; no rubric/cache changes.

Ordinary Ark requires the explicit version ID and exact /api/v3 endpoint.
Keep immutable scorer sources intact; record the adapter separately.
"""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

import rescore_glm305_comprehensive_grounded_20261009 as scorer

common=scorer.common
ENDPOINT='https://ark.cn-beijing.volces.com/api/v3/chat/completions'
EXPECTED=scorer.base.RESOLVED
ORIGINAL_SETUP=scorer.setup_worker

def probe_network():
    """Unauthenticated reachability check; no inference or key is sent."""
    opener,_=common.judge.request_transport(ENDPOINT)
    req=urllib.request.Request(ENDPOINT.rsplit('/',1)[0]+'/models',method='GET')
    try:
        with opener(req,timeout=8) as response:
            return {'reachable':True,'http_status':response.status}
    except urllib.error.HTTPError as exc:
        return {'reachable':True,'http_status':exc.code}
    except (OSError,urllib.error.URLError) as exc:
        return {'reachable':False,'error_type':type(exc).__name__}

def wait_network(force=False):
    """Share one probe across 64 processes; disconnected workers wait safely."""
    health=scorer.OUT/'NETWORK_STATUS.json'
    waited=False
    while True:
        with (scorer.OUT/'network.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=common.read(health) if health.exists() else {}
            now=time.time()
            if force or now>=state.get('next_probe_epoch',0):
                state={'at':common.now(),'checked_epoch':now,**probe_network()}
                state['next_probe_epoch']=time.time()+(20 if state['reachable'] else 30)
                state['state']='connected' if state['reachable'] else 'waiting_for_network'
                state['client_independent']=False
                common.save(health,state)
                force=False
            if state['reachable']:return waited
        waited=True
        time.sleep(5)

def request_model(model):
    if model not in (scorer.base.MODEL,EXPECTED):
        raise ValueError('Transport adapter cannot substitute a different judge')
    return EXPECTED

def setup_worker():
    ORIGINAL_SETUP()
    # Set the FULL endpoint: the frozen v1-oriented URL builder would otherwise
    # append /v1/chat/completions to /api/v3. Payload settings stay unchanged.
    os.environ['ANTHROPIC_BASE_URL']=ENDPOINT
    original_call=common.judge.call_judge
    def call_judge(prompt,model,**kwargs):
        model=request_model(model)
        wait_network()
        while True:
            try:
                return original_call(prompt,model,**kwargs)
            except common.judge.JudgeError as exc:
                # Keep provider/format/timeouts with a live route under the
                # frozen finite retry policy. Only a proven network outage
                # waits and resumes the same unresolved request automatically.
                if exc.category!='transport':raise
                if not wait_network(force=True):raise
    common.judge.call_judge=call_judge

def launch():
    jobs=scorer.prepare();out=scorer.OUT
    if not os.environ.get('LLM_API_KEY'):raise RuntimeError('LLM_API_KEY missing')
    # Actual frozen streaming client + route, not only a non-streaming curl.
    setup_worker()
    raw,meta=common.judge.call_judge('Return only the JSON object {"ok":true}.',EXPECTED,
        telemetry_path=out/'ark_standard_stream_probe.json',thinking_mode='disabled',
        max_output_tokens=512,timeouts=scorer.base.TIMEOUTS,
        api_protocol='chat_completions',output_schema=None)
    if meta.get('response_model')!=EXPECTED or meta.get('http_status')!=200:
        raise RuntimeError('Same-version streaming probe did not pass')
    if common.judge.decode_verdict(raw)!={'ok':True}:raise RuntimeError('Streaming JSON parse probe failed')
    protected={str(p):common.sha(p) for name in ('result.json','scan.json','review.json')
        for p in out.glob('cells/*/*/'+name)}
    counts=scorer.aggregate(jobs,'resuming_ark_standard')['counts']
    record={'at':common.now(),'endpoint':ENDPOINT,'requested_and_resolved_model':EXPECTED,
        'adapter':str(Path(__file__)),'adapter_sha256':common.sha(Path(__file__)),
        'stream_probe':{'http_status':meta['http_status'],'response_model':meta['response_model'],
            'thinking_chars':meta.get('thinking_chars'),'transport_route':meta.get('transport_route'),
            'request_url':meta.get('request_url'),'elapsed_seconds':meta.get('elapsed_seconds')},
        'concurrency_capacity':scorer.base.CAPACITY,'counts_at_resume':counts,
        'existing_cache_hashes':protected,'rubric_prompt_data_changed':False,
        'credentials_written':False,'client_sleep_independent_network':False,
        'network_note':'API plan changed, but this host still uses existing SSH proxy',
        'resume_policy':'first accepted results/scan/review immutable; unresolved only'}
    common.save(out/'ARK_STANDARD_RESUME_PROFILE.json',record)
    with (out/'worker.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,'-u',str(Path(__file__))],cwd=scorer.ROOT,
            env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    common.save(out/'launcher.json',{'at':common.now(),'pid':p.pid,'log':str(out/'worker.log'),
        'endpoint':ENDPOINT,'entrypoint':str(Path(__file__))})
    common.save(out/'API_RESUMED_ARK_STANDARD.json',{'at':common.now(),'state':'running',
        'pid':p.pid,'stream_probe_passed':True,'same_model_verified':True,
        'historical_pause_marker':'API_QUOTA_BLOCKED.json retained as history, not current worker state'})
    print(common.read(out/'launcher.json'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();os.umask(0o077)
    if args.launch:launch();return
    scorer.setup_worker=setup_worker
    sys.argv=[sys.argv[0]]
    scorer.base.OUT=scorer.OUT
    scorer.base.event('ark_standard_transport_resume',endpoint=ENDPOINT,
        requested_and_resolved_model=EXPECTED,concurrency_capacity=scorer.base.CAPACITY)
    scorer.main()
    # Verify cached accepted stages remained exactly untouched after completion.
    profile=common.read(scorer.OUT/'ARK_STANDARD_RESUME_PROFILE.json')
    changed=[p for p,h in profile['existing_cache_hashes'].items() if common.sha(p)!=h]
    if changed:raise RuntimeError('Accepted historical cache changed')
    common.save(scorer.OUT/'ARK_STANDARD_CACHE_PROTECTION_VERIFIED.json',
        {'at':common.now(),'unchanged':True,'files':len(profile['existing_cache_hashes'])})

if __name__=='__main__':main()
