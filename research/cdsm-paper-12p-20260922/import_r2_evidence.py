"""Copy verified completed experiment evidence into the portable paper bundle."""
from pathlib import Path
import hashlib,json,shutil

P=Path(__file__).resolve().parent
E=P.parent/'cdsm-review-strengthening-20260922';S=E/'server-results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
verification=json.loads((E/'LOCAL-VERIFICATION.json').read_text(encoding='utf-8'))
assert verification['passed'] and verification['summary_sha256']==sha(S/'SUMMARY.json')
T=P/'source-data/cdsm-review-strengthening-20260922';T.mkdir(exist_ok=True)
relative=['SUMMARY.json','AUDIT.json','COMPLETION.json','PLAN.json','PROTOCOL.zh-CN.md','DEPENDENCIES.json',
          'FROZEN.json','SELECTION.json','SELECTION-FROZEN.json','ENVIRONMENT.json','INPUTS-VERIFIED.json',
          'strengthen.h','bench.cpp','check.cpp','compile.sh','run_server.py','analyze.py','preflight.py',
          'engineering/PASS.json','engineering/LEGACY-PARITY.json']
relative += [p.relative_to(S).as_posix() for d in ['vendor','queries']for p in (S/d).rglob('*')if p.is_file()]
for rel in relative:
    p=S/rel;q=T/rel;q.parent.mkdir(parents=True,exist_ok=True)
    if q.exists():assert sha(q)==sha(p)
    else:shutil.copy2(p,q)
for rel in ['LOCAL-VERIFICATION.json','SERVER-BUNDLE.json','RUNBOOK.zh-CN.md','verify_bundle.py']:
    shutil.copy2(E/rel,T/rel)
for rel in ['SUMMARY.json','AUDIT.json','COMPLETION.json','REPORT.zh-CN.md']:
    q=E/rel
    if q.exists():assert sha(q)==sha(S/rel)
    else:shutil.copy2(S/rel,q)

# The shared execution controller is a small dependency, unlike the server's
# static Faiss library, vector files and graph indexes.
engine=S/'archive-dependencies/resource_runner.py'
identity=json.loads((S/'FROZEN.json').read_text())['sources']
assert sha(engine)==identity['research/hnsw-adaptive-baselines-20260920/resource_runner.py']
target=P/'source-data/hnsw-adaptive-baselines-20260920/resource_runner.py'
if target.exists():assert sha(target)==sha(engine)
else:shutil.copy2(engine,target)
print(json.dumps({'portable_evidence_files':len(list(T.rglob('*'))),'summary_sha256':sha(T/'SUMMARY.json'),'raw_archive_separate':True}))
