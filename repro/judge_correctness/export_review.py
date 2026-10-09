#!/usr/bin/env python3
"""Export fixed instructions/schema directly from published scorer code."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/evaluation'))
import rescore_glm305_correctness_20261009 as base
import rescore_glm305_comprehensive_grounded_20261009 as r4

def main():
    out=ROOT/'docs/judge_correctness_20261009/prompts';out.mkdir(parents=True,exist_ok=True)
    primary=base.SCAN;primary_schema=base.scan_schema()
    # Exact same local setup used by workers; no request is made here.
    os.environ.setdefault('ARK_SCORING_PROXY','http://127.0.0.1:38790')
    r4.setup_worker()
    files={'01_primary_scan.txt':primary,'02_comprehensive_scan.txt':r4.AUDIT,
        '03_effective_review.txt':base.REVIEW,
        '01_primary_schema.json':json.dumps(primary_schema,ensure_ascii=False,indent=2)+'\n',
        '02_comprehensive_schema.json':json.dumps(r4.evidence_schema(),ensure_ascii=False,indent=2)+'\n',
        '03_review_schema.json':json.dumps(r4.ORIGINAL_REVIEW_SCHEMA(),ensure_ascii=False,indent=2)+'\n'}
    for name,text in files.items():(out/name).write_text(text)
    source_files=[ROOT/'code/evaluation'/x for x in (
        'rescore_advisory263_pro_20261007.py','rescore_glm305_correctness_20261009.py',
        'rescore_glm305_comprehensive_grounded_20261009.py','resume_glm305_ark_standard_20261009.py')]
    manifest={'kind':'fixed-instruction-export-not-live-request',
        'dynamic_input':'one complete question, original applicable rubric, one frozen final answer with E/Q evidence IDs; never arm/model/old score',
        'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list(out.glob('*'))+source_files if p.is_file()}}
    (out.parent/'prompt_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Exported exact fixed instructions and schemas; no API call.')

if __name__=='__main__':main()
