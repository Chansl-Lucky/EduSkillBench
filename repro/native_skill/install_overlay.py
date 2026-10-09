"""Restore only the eight documented patched BenchFlow files into a task venv."""
import csv
import hashlib
from pathlib import Path
import shutil
import sys
import benchflow

def main():
    if sys.prefix == sys.base_prefix: raise SystemExit('Use an independent virtualenv, not system Python')
    package = Path(benchflow.__file__).resolve().parent.parent
    assets = Path(__file__).parent / 'environment'
    records = list(csv.DictReader((assets / 'framework.csv').open()))
    for row in records:
        src = assets / 'overlay' / row['path']
        if not src.exists(): continue
        assert hashlib.sha256(src.read_bytes()).hexdigest() == row['upstream_sha256']
        dst = package / row['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print('Restored pinned BenchFlow overlay in independent virtualenv')

if __name__ == '__main__': main()

