"""Read-only version/source/image preflight, never calls a model API."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import benchflow

def main():
    assert sys.version_info[:2] == (3, 12)
    assert importlib.metadata.version('benchflow') == '0.6.7'
    package = Path(benchflow.__file__).resolve().parent.parent
    records = list(csv.DictReader((Path(__file__).parent / 'environment/framework.csv').open()))
    for row in records:
        assert hashlib.sha256((package / row['path']).read_bytes()).hexdigest() == row['upstream_sha256'], row['path']
    source = Path(__file__).resolve().parents[2] / 'data/exports/eduskillbench-305-20261003/cases.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'ab0eeee5246679373d7ceb23bd87f80630c87698150d0fc49085ec7fc369f777'
    subprocess.run(['docker','run','--rm','eduskillbench-opencode-runtime:1.18.11-rg1','sh','-c',
        'test "$(/opt/benchflow/node/bin/node --version)" = v22.20.0 && test "$(/opt/benchflow/js-agents/bin/opencode --version)" = 1.18.11'], check=True)
    print(json.dumps({'benchflow': '0.6.7', 'matched_framework_files': len(records), 'cases': 305, 'node': '22.20.0', 'opencode': '1.18.11'}))

if __name__ == '__main__': main()

