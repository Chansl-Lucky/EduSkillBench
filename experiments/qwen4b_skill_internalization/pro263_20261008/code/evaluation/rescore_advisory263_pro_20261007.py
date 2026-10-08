#!/usr/bin/env python3
"""Rescore frozen parent final answers; pinned upstream prompt/validation/transport.

No generation, training, Docker changes or old-score overwrites. Each successful
cell is immutable and bound to its exact case, answer and scoring configuration.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'artifacts/eval305_handoff_sft246_20261004_r2'
UP = ROOT / 'artifacts/sft246_grpo_20261005_r2/api_switch_20261007/upstream_scoring'
OUT = ROOT / 'artifacts/rescore_advisory263_pro_20261007'
PROFILE = ROOT / 'configs/evaluation/upstream_pro_20261007.json'
ARMS = ('base', 'base_skill', 'sft')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

judge = module('pinned_pro_judge', UP/'repro/judge.py')
native = module('pinned_native_protocol', UP/'repro/source_protocol.py')

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n'); tmp.replace(p)

def append(p, x):
    with Path(p).open('a') as f:
        f.write(json.dumps(x, ensure_ascii=False)+'\n')

def event(name, **kw):
    x = {'at': now(), 'event': name, **kw}
    append(OUT/'events.jsonl', x); print(json.dumps(x, ensure_ascii=False), flush=True)

def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def parse_data(row):
    data = row.get('data', {})
    return json.loads(data) if isinstance(data, str) else data

def extract_final(db):
    """Only the latest assistant message in the user-task parent session.

    No child text, tool return, duplicated ACP event, earlier planning message,
    or generated title is used as an answer. Multiple root sessions fail closed.
    """
    roots = {r['id'] for r in db.get('session', []) if not r.get('parent_id')}
    task_roots = {m.get('session_id') for m in db.get('message', [])
                  if m.get('session_id') in roots and parse_data(m).get('role') == 'user'}
    if len(task_roots) != 1:
        return '', {'terminal': False, 'reason': 'ambiguous_or_missing_task_parent', 'root_ids': sorted(task_roots)}
    root = next(iter(task_roots))
    messages = [(m, parse_data(m)) for m in db.get('message', [])
                if m.get('session_id') == root and parse_data(m).get('role') == 'assistant']
    messages.sort(key=lambda x: (x[0].get('time_created', 0), x[0]['id']))
    if not messages:
        return '', {'terminal': False, 'reason': 'missing_parent_assistant', 'session_id': root}
    row, message = messages[-1]
    parts = [r for r in db.get('part', []) if r.get('message_id') == row['id']
             and parse_data(r).get('type') == 'text' and not parse_data(r).get('synthetic')]
    parts.sort(key=lambda r: (r.get('time_created', 0), r['id']))
    answer = '\n'.join(parse_data(r).get('text', '') for r in parts).strip()
    terminal = bool(answer and message.get('finish') in ('stop', 'end_turn'))
    return answer, {'terminal': terminal, 'session_id': root, 'message_id': row['id'],
                    'finish': message.get('finish'), 'reason': None if terminal else 'no_terminal_parent_answer',
                    'part_ids': [r['id'] for r in parts]}

def top_share(case, verdict):
    active = {r['id']: r for r in case['criteria'] if r['id'] in case['applicable_ids']}
    items = {r['id']: r for r in verdict['items']}
    if not active or items.keys() != active.keys():
        raise ValueError('incomplete native dimensions')
    return sum(items[k]['level'] == native.labels(c)[0] for k, c in active.items()) / len(active)

def prepare():
    profile = read(PROFILE)
    if profile['evaluation_suite'] != 'advisory' or profile['expected_tasks_per_arm'] != 263:
        raise ValueError('only advisory263 is authorized')
    cases = [c for c in read(OLD/'cases.json') if c['suite'] == 'advisory']
    fresh = {c['task_id']: c for c in read(UP/'data/exports/eduskillbench-305-20261003/cases.json') if c['suite'] == 'advisory'}
    assert len(cases) == 263 and {c['task_id']: c for c in cases} == fresh
    inputs = {'source_cases': sha(OLD/'cases.json'), 'rollout_ledger': sha(OLD/'rollouts.jsonl'),
              'old_flash_scores': sha(OLD/'scores.jsonl'), 'profile': sha(PROFILE),
              'upstream_judge': sha(UP/'repro/judge.py'), 'upstream_native': sha(UP/'repro/source_protocol.py'),
              'runner': sha(Path(__file__))}
    if (OUT/'manifest.json').exists():
        old_inputs=read(OUT/'manifest.json')['input_hashes']
        assert {k:v for k,v in old_inputs.items() if k!='runner'} == {k:v for k,v in inputs.items() if k!='runner'}, 'frozen scoring inputs changed'
        if old_inputs['runner']!=inputs['runner']:
            revision=read(OUT/'runner_revision.json')
            assert revision['original_runner_sha256']==old_inputs['runner'] and revision['current_runner_sha256']==inputs['runner'], 'unauthorized runner change'
        return read(OUT/'inputs.json')
    ledger = {(x['arm'], x['task_id']): x for x in rows(OLD/'rollouts.jsonl')}
    jobs = []
    for case in cases:
        for arm in ARMS:
            record = ledger[(arm, case['task_id'])]
            folder = OUT/'cells'/arm/case['task_id']
            db_path = Path(record['job'])/'capture/opencode_database.json'
            if not db_path.exists():
                db_path = Path(record['job'])/'agent/opencode_database.json'
            answer, extraction = extract_final(read(db_path)) if db_path.exists() else ('', {'terminal': False, 'reason': 'missing_database'})
            category = 'execution_failure' if record.get('error') else ('answer_extraction_failure' if not extraction['terminal'] else None)
            x = {'arm': arm, 'task_id': case['task_id'], 'case': case, 'answer': answer,
                 'job': record['job'], 'source_error': record.get('error'), 'category': category,
                 'extraction': extraction, 'source_database': str(db_path),
                 'source_database_sha256': sha(db_path) if db_path.exists() else None}
            x['binding'] = digest({'case': case, 'answer': answer, 'profile': profile, 'extraction': extraction})
            save(folder/'input.json', x)
            (folder/'answer.md').write_text(answer+'\n')
            if category:
                save(folder/'result.json', {'arm': arm, 'task_id': case['task_id'], 'binding': x['binding'],
                     'status': category, 'source_error': record.get('error'), 'extraction': extraction,
                     'report_score': 0.0, 'failure_zero': True, 'score_origin': 'failure_policy_not_judge'})
            jobs.append(x)
    save(OUT/'inputs.json', jobs)
    save(OUT/'manifest.json', {'at': now(), 'input_hashes': inputs, 'profile': profile,
         'arms': ARMS, 'tasks_per_arm': 263, 'cells': 789, 'judge_targets': sum(not x['category'] for x in jobs),
         'answer_policy': 'strict parent final answer only, no rollout regeneration',
         'old_flash_results_protected': True, 'generation_environment': 'existing Docker/OpenCode, NOT upstream direct Chat',
         'score_comparison_caveat': 'Judge AND evidence scope differ from previous all-trajectory Flash scores',
         'process_workers': '8 -> 32 -> 64 gated stress test; fallback on rate limits'})
    event('prepared', cells=len(jobs), judge_targets=sum(not x['category'] for x in jobs),
          exclusions=dict(Counter(x['category'] for x in jobs if x['category'])))
    return jobs

def score_one(x):
    folder=OUT/'cells'/x['arm']/x['task_id']
    with (folder/'.cell.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        return score_locked(x)

def score_locked(x):
    profile = read(PROFILE); folder = OUT/'cells'/x['arm']/x['task_id']; cache = folder/'result.json'
    if cache.exists():
        saved = read(cache)
        assert saved['binding'] == x['binding'], 'cell inputs changed'
        if saved['status'] == 'evaluated':
            native.validate(saved['raw_verdict'], x['case'], x['answer'], require_counterevidence=True)
            return saved
    text = native.prompt(x['case'], x['answer']); request_text = text
    active = [c for c in x['case']['criteria'] if c['id'] in x['case']['applicable_ids']]
    began = time.monotonic()
    for attempt in range(2):
        stamp = str(time.time_ns())
        try:
            raw, meta = judge.call_judge(request_text, profile['judge_model'],
                telemetry_path=folder/f'trace_{stamp}.json', thinking_mode='disabled',
                max_output_tokens=10000, timeouts=profile['timeouts'], api_protocol='chat_completions',
                output_schema=native.schema(active))
            (folder/f'raw_{stamp}.txt').write_text(raw); save(folder/f'response_{stamp}.json', meta)
            if meta.get('response_model') != profile['availability_probe_returned_model']:
                raise RuntimeError('Judge model resolution changed: '+str(meta.get('response_model')))
            if meta.get('stop_reason') not in ('end_turn', 'stop_sequence'):
                raise judge.JudgeError('output_truncated_or_incomplete', False, meta)
            verdict = native.validate(judge.decode_verdict(raw), x['case'], x['answer'], require_counterevidence=True)
            result = {'arm': x['arm'], 'task_id': x['task_id'], 'binding': x['binding'],
                      'status': 'evaluated', 'judge_requested': profile['judge_model'],
                      'judge_model': meta['response_model'], 'raw_verdict': judge.decode_verdict(raw),
                      'verdict': verdict, 'report_score': top_share(x['case'], verdict),
                      'strict_pass': all(i['level'] == native.labels(c)[0] for i, c in zip(verdict['items'], active)),
                      'failure_zero': False, 'metadata': meta, 'elapsed_seconds': round(time.monotonic()-began, 2)}
            save(cache, result); return result
        except Exception as exc:
            meta = getattr(exc, 'metadata', {})
            category = getattr(exc, 'category', type(exc).__name__)
            error = re.sub(r'(?:ark-|sk-)[A-Za-z0-9_-]+', '[REDACTED]', str(exc))[:500]
            failure = {'arm': x['arm'], 'task_id': x['task_id'], 'binding': x['binding'],
                       'status': 'judge_failed', 'category': category, 'error': error,
                       'metadata': meta, 'http_status': meta.get('http_status'), 'attempt': attempt+1,
                       'at': now(), 'elapsed_seconds': round(time.monotonic()-began, 2)}
            save(folder/f'failure_{stamp}.json', failure)
            # Keep two-attempt validation-feedback rule. Never rapid-retry 429,
            # authentication, model changes or the upstream total deadline.
            if isinstance(exc, RuntimeError) or meta.get('http_status') in (401,403,429) or meta.get('timeout_kind')=='total_deadline' or (isinstance(exc, judge.JudgeError) and not exc.retryable) or attempt == 1:
                save(folder/'error.json', failure); return failure
            request_text = text+'\n上次输出未通过程序校验：'+error+'。请重新评审并返回完整 JSON，逐项检查编号、等级及证据编号；不要修改评分标准或为了通过校验提高等级。'
            time.sleep(1)
    raise AssertionError('unreachable')

def summarize(jobs, phase, concurrency=0):
    totals = {}; pending = 0; failures = Counter()
    for arm in ARMS:
        scores = []; valid_scores = []; n = 0; executed = 0; errs = 0
        for x in jobs:
            if x['arm'] != arm: continue
            folder = OUT/'cells'/arm/x['task_id']; p = folder/'result.json'
            if p.exists():
                r = read(p); scores.append(r['report_score']); n += r['status']=='evaluated'; executed += r['failure_zero']
                if r['status']=='evaluated':valid_scores.append(r['report_score'])
            elif (folder/'error.json').exists():
                errs += 1; failures[read(folder/'error.json')['category']] += 1
            else: pending += 1
        complete = n+executed+errs == 263
        totals[arm] = {'target':263, 'evaluated':n, 'execution_or_extraction_failures':executed,
                      'judge_unresolved':errs, 'valid_score_mean_secondary':statistics.mean(valid_scores) if valid_scores else None,
                      'fixed_denominator_score':sum(scores)/263 if complete else None,
                      'failure_zero_note':'unresolved final judge cells count zero only at terminal reporting'}
    state = {'at':now(), 'phase':phase, 'concurrency':concurrency, 'by_arm':totals,
             'pending':pending, 'judge_error_categories':dict(failures)}
    save(OUT/'status.json', state); return state

def paper_report(jobs):
    """Same task-macro and paired source-cluster bootstrap as pinned build_results.

    Only the three local arms differ. Unresolved terminal scores are zero,
    retaining their category. No arbitrary ordinal-to-number conversion.
    """
    import numpy as np
    records=[]
    for x in jobs:
        folder=OUT/'cells'/x['arm']/x['task_id']
        r=read(folder/'result.json') if (folder/'result.json').exists() else read(folder/'error.json')
        good=r['status']=='evaluated'
        records.append({'arm':x['arm'],'task_id':x['task_id'],'skill':x['task_id'].split('__')[0],
                        'source_group':x['case']['original_source'],'status':r['status'],
                        'score':r['report_score'] if good else 0.,'failure_zero':not good,
                        'strict_pass':bool(r.get('strict_pass')) if good else False,
                        'review_flag':any(i.get('counterevidence') for i in r.get('verdict',{}).get('items',[])),
                        'category':r.get('category',r['status']) if not good else None})
    ids=sorted({x['task_id'] for x in jobs});assert len(ids)==263
    lookup={(r['arm'],r['task_id']):r for r in records}
    clusters=sorted({lookup['base',t]['source_group'] for t in ids});assert len(clusters)==30
    ixgroups=[np.array([i for i,t in enumerate(ids) if lookup['base',t]['source_group']==g]) for g in clusters]
    rng=np.random.default_rng(20261005);pairs={}
    for before,after in [('base','base_skill'),('base','sft'),('base_skill','sft')]:
        b=[lookup[before,t] for t in ids];w=[lookup[after,t] for t in ids]
        bv=np.array([r['score'] for r in b]);wv=np.array([r['score'] for r in w]);delta=wv-bv
        boot=[]
        for _ in range(5000):
            ix=np.concatenate([ixgroups[i] for i in rng.integers(0,len(clusters),len(clusters))])
            boot.append(float(np.mean(delta[ix])))
        paired=[i for i in range(263) if not b[i]['failure_zero'] and not w[i]['failure_zero']]
        unflagged=[i for i in paired if not b[i]['review_flag'] and not w[i]['review_flag']]
        pairs[before+'__'+after]={'n':263,'baseline':float(np.mean(bv)),'comparison':float(np.mean(wv)),
              'delta':float(np.mean(delta)),'gain':float(np.mean(delta)/(1-np.mean(bv))) if np.mean(bv)<1 else None,
              'ci':list(map(float,np.quantile(boot,[.025,.975]))),
              'wins_ties_losses':[int(np.sum(delta>1e-9)),int(np.sum(np.abs(delta)<=1e-9)),int(np.sum(delta< -1e-9))],
              'complete_pair_n':len(paired),'complete_pair_delta':float(np.mean(delta[paired])) if paired else None,
              'unflagged_pair_n':len(unflagged),'unflagged_pair_delta':float(np.mean(delta[unflagged])) if unflagged else None,
              'cluster_count':len(clusters)}
    skills=[]
    for arm in ARMS:
        for sid in sorted({r['skill'] for r in records}):
            subset=[r for r in records if r['arm']==arm and r['skill']==sid]
            skills.append({'arm':arm,'skill':sid,'n':len(subset),'mean':statistics.mean(r['score'] for r in subset),
                           'failure_count':sum(r['failure_zero'] for r in subset)})
    result={'at':now(),'metric':'per-task highest-native-label share, macro over 263, terminal failures zero',
            'bootstrap':'5000 paired source-cluster resamples, 30 sources, seed 20261005; conditional on stored outcomes',
            'pairs':pairs,'skill_results':skills,'arms':{},'generation_environment':'existing Docker/OpenCode, not upstream direct Chat',
            'comparison_caveat':'Pro final-answer scores must not be mixed with old Flash full-trajectory scores'}
    for arm in ARMS:
        rs=[r for r in records if r['arm']==arm]
        result['arms'][arm]={'n':263,'mean':statistics.mean(r['score'] for r in rs),
                             'strict_pass':statistics.mean(r['strict_pass'] for r in rs),
                             'failure_count':sum(r['failure_zero'] for r in rs),
                             'review_flags':sum(r['review_flag'] for r in rs)}
    save(OUT/'paper_report.json',result);save(OUT/'paper_rows.json',records)
    lines=['# 263 题统一 Pro 重评分','',f'更新时间：{now()}。失败计零、固定分母 263；旧 Flash 不混入。','',
           '| 版本 | 最高档维度占比 | 全维最高档比例 | 失败数 |','| --- | ---: | ---: | ---: |']
    for arm,r in result['arms'].items():
        lines.append(f"| {arm} | {100*r['mean']:.2f}% | {100*r['strict_pass']:.2f}% | {r['failure_count']} |")
    lines+=['','生成环境仍是旧 Docker/OpenCode；本次对齐评分，不等于对齐直接 Chat 生成。',
            '逐项等级、证据、反证、原始响应见 cells；配对来源簇区间与 Skill 分项见 paper_report.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def batch(targets, concurrency, jobs, stage):
    event('batch_start', stage=stage, concurrency=concurrency, targets=len(targets))
    outputs = []; begun = time.monotonic()
    with ProcessPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(score_one,x):(x['arm'],x['task_id']) for x in targets}
        for f in as_completed(futures):
            r=f.result(); outputs.append(r)
            append(OUT/'scores.jsonl', r)
            event('cell_done', arm=r['arm'], task_id=r['task_id'], status=r['status'], score=r.get('report_score'), http_status=r.get('http_status'))
            summarize(jobs, stage, concurrency)
    elapsed = time.monotonic()-begun
    pressure = {'at':now(), 'stage':stage, 'concurrency':concurrency, 'targets':len(targets),
                'accepted':sum(r['status']=='evaluated' for r in outputs),
                'http_429':sum(r.get('http_status')==429 for r in outputs),
                'model_identity_failures':sum(r.get('category')=='RuntimeError' for r in outputs),
                'capacity_failures':sum(r.get('category') in ('transport','http_error','provider_stream_error') for r in outputs),
                'elapsed_seconds':round(elapsed,2),
                'cell_latency_median_seconds':statistics.median(r['elapsed_seconds'] for r in outputs) if outputs else None}
    append(OUT/'pressure_tests.jsonl',pressure); event('batch_done',**{k:v for k,v in pressure.items() if k!='at'})
    return pressure

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    guard=(OUT/'run.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=prepare()
    if args.prepare_only:
        print(json.dumps(summarize(jobs,'prepared'),ensure_ascii=False));return
    if not os.environ.get('LLM_API_KEY'):raise RuntimeError('missing LLM_API_KEY')
    os.environ['ANTHROPIC_BASE_URL']='https://ark.cn-beijing.volces.com/api/plan/v1'
    os.environ.pop('ANTHROPIC_API_KEY',None);os.environ.pop('ANTHROPIC_AUTH_TOKEN',None)
    def todo():
        return [x for x in jobs if not x['category'] and not (OUT/'cells'/x['arm']/x['task_id']/'result.json').exists()]
    concurrency=8
    for proposed in (8,32,64):
        pending=todo()
        if not pending:break
        pressure=batch(pending[:proposed],proposed,jobs,'pressure_'+str(proposed))
        if pressure['model_identity_failures']:
            summarize(jobs,'blocked_model_identity',0);return
        concurrency=proposed
        if pressure['http_429'] or pressure['capacity_failures']/pressure['targets'] > .10:
            concurrency=max(8,proposed//2);event('pressure_fallback',to=concurrency)
            if pressure['http_429']:time.sleep(30)
            break
    event('formal_start',concurrency=concurrency,remaining=len(todo()))
    # Bounded blocks allow concurrency backoff rather than queueing all 789.
    for wave in range(2):
        pending=todo()
        while pending:
            targets,pending=pending[:concurrency],pending[concurrency:]
            pressure=batch(targets,concurrency,jobs,'formal_wave_'+str(wave))
            if pressure['model_identity_failures']:
                summarize(jobs,'blocked_model_identity',0);return
            if pressure['http_429']:
                concurrency=max(8,concurrency//2);event('rate_limit_cooldown',concurrency=concurrency);time.sleep(30)
        if not todo():break
    state=summarize(jobs,'finished' if not todo() else 'finished_with_judge_failures',0)
    paper_report(jobs)
    save(OUT/'FINISHED.json',state);event('finished',remaining=len(todo()))
    assert sha(OLD/'scores.jsonl')==read(OUT/'manifest.json')['input_hashes']['old_flash_scores']

if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        save(OUT/'fatal_error.json',{'at':now(),'type':type(exc).__name__,'error':str(exc)[:1000]})
        raise
