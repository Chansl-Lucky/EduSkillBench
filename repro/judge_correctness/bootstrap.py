#!/usr/bin/env python3
"""Prepare byte-identical import dependencies and one public regression fixture.

No API calls, credentials, scoring changes or live-result overwrites.
"""
from pathlib import Path
import hashlib
import shutil

ROOT=Path(__file__).resolve().parents[2]
UP=ROOT/'artifacts/sft246_grpo_20261005_r2/api_switch_20261007/upstream_scoring/repro'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def copy_once(source,dest,expected=None):
    if expected and sha(source)!=expected:raise RuntimeError('Pinned source differs: '+str(source))
    if dest.exists():
        if sha(dest)!=sha(source):raise RuntimeError('Refusing to overwrite existing runtime file: '+str(dest))
    else:
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)

def main():
    pinned={'judge.py':'dc61d528ed68c0c075af6106a25ce2e1e7e7e6848d9ec541e44412e5ed7deacf',
        'source_protocol.py':'3e5c9122789a7090e5ce847cc47a664838f99d5cbbbc1f794e86fd7e4a8202aa'}
    for name,digest in pinned.items():
        copy_once(ROOT/'repro/native_skill/upstream'/name,UP/name,digest)
    fixture=ROOT/'repro/judge_correctness/fixtures/vector_diagnostic'
    target=ROOT/'artifacts/glm305_recovery_20261009_r1/cells/base/lesson-builder__cn24_09'
    for name in ('input.json','result.json'):copy_once(fixture/name,target/name)
    print('Pinned imports and public regression fixture verified; no inference started.')

if __name__=='__main__':main()
