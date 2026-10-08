#!/usr/bin/env python3
"""Publish only reviewed experiment code/config/score reports to the user's repo.

Uses an isolated clone: no staging/resetting the original dirty worktree.
Secrets are read from the process environment and never included in the packet.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]
TARGET='https://github.com/Chansl-Lucky/EduSkillBench.git'
REL=Path('experiments/qwen4b_skill_internalization/pro263_20261008')

def run(argv,env=None,**kwargs):
    return subprocess.run(argv,check=True,capture_output=True,text=True,env=env,**kwargs)

def token():
    if os.environ.get('GH_TOKEN'):return os.environ['GH_TOKEN']
    password=os.environ.get('PUBLISH_SUDO_PASSWORD')
    if not password:raise RuntimeError('GitHub credentials unavailable: provide GH_TOKEN or authorized credential access')
    r=run(['sudo','-S','-u','gpuuser','/home/gpuuser/csl/bin/gh','auth','token'],input=password+'\n',timeout=20)
    value=r.stdout.strip()
    if not value:raise RuntimeError('GitHub CLI returned no credential')
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--packet-only',action='store_true');a=p.parse_args()
    out=a.output.resolve();report=out/'comparison'
    if not (out/'FINISHED.json').exists() or not (report/'paper_report.json').exists():raise RuntimeError('Evaluation/report incomplete; publication refused')
    packet=out/'github_packet';packet.mkdir(exist_ok=True)
    for name in ('REPORT.md','paper_report.json','paper_rows.json'):
        shutil.copy2(report/name,packet/name)
    stats=json.loads((packet/'paper_report.json').read_text())
    extra=['','## GRPO 对照差值','',
           '| 对照 | GRPO − 对照（百分点） | 95%来源簇区间（百分点） | 双方有效配对题数 |',
           '| --- | ---: | --- | ---: |']
    for arm in ('base','base_skill','sft'):
        x=stats['pairs'][arm+'__grpo'];lo,hi=x['ci']
        extra.append(f"| {arm} | {100*x['delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {x['complete_pair_n']} |")
    extra+=['','区间跨 0 时不声称统计显著提升。该区间条件于本次保存结果，不代表重复生成随机性的区间。',
            '旧三组尚有14份Judge校验失败，报告中按失败计零并保留；不能把它们误说成模型全部答错。',
            '本表仅263题，覆盖9个Skill；不含42题，不声称覆盖全部14个训练Skill。']
    with (packet/'REPORT.md').open('a') as f:f.write('\n'.join(extra)+'\n')
    for name in ('manifest.json','FINISHED.json','SMOKE_GATE.json'):
        shutil.copy2(out/name,packet/name)
    shutil.copy2(ROOT/'artifacts/rescore_advisory263_pro_20261007/status.json',packet/'previous_pro_scoring_status.json')
    files=['code/evaluation/eval_grpo263_pro_20261008.py','code/evaluation/rescore_advisory263_pro_20261007.py',
           'code/evaluation/test_rescore_advisory263_pro_20261007.py','code/evaluation/publish_grpo263_20261008.py',
           'code/rlvr/eval305_handoff_20261004.py','code/rlvr/eval305_20261004.py',
           'code/rlvr/train_sft_grpo_20261005.py','code/rlvr/sft_grpo_reward.py','code/rlvr/sft_grpo_checkpoint.py','code/rlvr/judge.py',
           'configs/evaluation/upstream_pro_20261007.json','docs/PRO263_RESCORING_RUN_2026-10-07.md',
           'docs/SCORING_ALIGNMENT_2026-10-07.md','docs/GRPO263_EVALUATION_2026-10-08.md']
    for name in files:
        dest=packet/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    # Native scoring source is pinned; preserve the exact reused implementations.
    up=ROOT/'artifacts/sft246_grpo_20261005_r2/api_switch_20261007/upstream_scoring'
    for name in ('repro/judge.py','repro/source_protocol.py','docs/scoring/judge_model.json'):
        dest=packet/'pinned_upstream'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(up/name,dest)
    readme='''# Qwen3-4B Skill internalization: 263-task Pro evaluation

See [REPORT.md](REPORT.md) for the four-arm result table, [paper_report.json](paper_report.json)
for paired source-cluster intervals and per-Skill results, and [paper_rows.json](paper_rows.json)
for all task-level outcomes. Failed execution/Judge cells remain in the fixed denominator;
they are not removed or replaced by the best repeat. Inspect failure counts before interpreting gains.

## Experiment

- Base: original Qwen3-4B-Instruct-2507, no Skill.
- Base+Skill: original model, matched Skill and accessible resources.
- SFT: LoRA distilled from the accepted 246-task training release, no Skill.
- GRPO: 123 outcome-based optimizer steps starting from that SFT adapter, no Skill.
- Test: only the 263 advisory tasks; 9 Skills and 30 source documents (not all 14 training Skills).
- Generation: existing BenchFlow/Docker/OpenCode 1.18.11, education-single-turn system,
  600-second task timeout. GRPO replaces only the trained adapter/alias and server port.
- Judge: requested deepseek-v4-pro, resolved deepseek-v4-pro-ga-260813; Ark Plan Chat,
  thinking disabled, max output 10000, original streaming deadlines, no explicit temperature/seed.
- Scoring: pinned upstream commit 6270c70e793ad625297dcce8074bc5cd601aa5c1,
  source-native-v1 prompt and validation; parent final answer only.
- Main statistic: highest-native-label fraction per task, then macro mean over 263;
  terminal failures zero. 5000 paired source-cluster bootstrap resamples, seed 20261005.

The upstream formal experiment generated answers via direct Chat. Our Docker/OpenCode
generation is still different. Scoring alignment must not be described as full environment equality.
Old Flash scores used full multi-agent trajectories and are not mixed into this Pro table.
The previous three-arm Pro run retained unresolved Judge validation failures; report them explicitly.

## Run on the experiment server

The launch documentation and original manifests are included. Local model weights,
Docker jobs, API keys, account credentials and raw request logs are deliberately excluded.
Use separate environments: the existing .venv-eval305-handoff for BenchFlow,
.venv-qwen4b-serve for vLLM, and .venv for training/reporting. Do not upgrade their dependencies
mid-comparison. Restore pinned upstream files under the documented snapshot paths and
provide the Qwen model + completed adapter locally.

```bash
export PYTHONPATH="$PWD/code:$PWD/code/evaluation"
export LLM_API_KEY='YOUR_ARK_PLAN_KEY'
.venv-eval305-handoff/bin/python code/evaluation/eval_grpo263_pro_20261008.py
```

This evaluation only generates GRPO answers; no Base/SFT reroll or training occurs.
Generation concurrency is 8, Judge capacity 64; CPU-only scoring overlaps generation.
The model server is terminated on completion/error. Existing successful cells are cached.
The publication helper uses an isolated clone of the user's repository and a secrets scan.
These scripts reflect a server handoff, not a self-contained one-command install without models/assets.
'''
    (packet/'README.md').write_text(readme)
    provenance={}
    for f in packet.rglob('*'):
        if not f.is_file():continue
        text=f.read_text()
        if re.search(r'\b(?:ark-[a-zA-Z0-9-]{16,}|sk-[a-zA-Z0-9_-]{16,}|gh[pousr]_[a-zA-Z0-9]{20,}|github_pat_[a-zA-Z0-9_]{20,})',text):
            raise RuntimeError('Secret-like value detected in publication file: '+str(f.relative_to(packet)))
        text=text.replace(str(ROOT),'${EDUSKILL_ROOT}').replace('${MODEL_ROOT}','${MODEL_ROOT}')
        f.write_text(text)
        provenance[str(f.relative_to(packet))]=hashlib.sha256(f.read_bytes()).hexdigest()
    (packet/'file_hashes.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if a.packet_only:print(json.dumps({'packet':str(packet),'files':len(provenance)}));return
    secret=token();auth=base64.b64encode(('x-access-token:'+secret).encode()).decode()
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+auth)
    clone=Path(tempfile.mkdtemp(prefix='eduskill-pro263-publish-',dir=str(out)))
    run(['git','clone','--depth','1',TARGET,str(clone)],env=env,timeout=180)
    dest=clone/REL;dest.mkdir(parents=True,exist_ok=True)
    shutil.copytree(packet,dest,dirs_exist_ok=True)
    run(['git','-C',str(clone),'add','--',str(REL)],env=env,timeout=20)
    # Stage only the experiment packet. Existing upstream/user content stays untouched.
    staged=run(['git','-C',str(clone),'diff','--cached','--name-only'],env=env).stdout.splitlines()
    assert staged and all(x.startswith(str(REL)+'/') for x in staged)
    run(['git','-C',str(clone),'-c','user.name=Chansl-Lucky','-c','user.email=Chansl-Lucky@users.noreply.github.com',
         'commit','-m','Report Qwen4B SFT-GRPO advisory263 evaluation with aligned Pro scoring'],env=env,timeout=40)
    commit=run(['git','-C',str(clone),'rev-parse','HEAD'],env=env).stdout.strip()
    run(['git','-C',str(clone),'push','origin','HEAD:main'],env=env,timeout=180)
    result={'repository':TARGET,'commit':commit,'url':'https://github.com/Chansl-Lucky/EduSkillBench/tree/main/'+str(REL),'files':len(staged)}
    (out/'PUBLISHED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
