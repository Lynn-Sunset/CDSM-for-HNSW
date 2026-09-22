"""Retain historical parameter evidence without mixing it with C++ results."""
from pathlib import Path
import hashlib,json

R=Path(__file__).resolve().parent
S=R/'source-data/depth-allocation-20260910/DEVELOPMENT-RESULTS.json'
data=json.loads(S.read_text(encoding='utf-8'))['datasets']['t2i']
rows=[]
for name in ['fast-one-100','fast-one-400','fast-one-1600','fast-four-100','fast-four-400','fast-four-1600']:
    x=next(x for x in data['summary']if x['policy']==name)
    routes=1 if '-one-'in name else 4
    w=x['mean_counters']
    assert abs(w['dc']-11208.054833333334)<1e-8
    rows.append(dict(policy=name,routes=routes,quota=int(name.split('-')[-1]),
        mean_scout_dc=w['scout_dc'],quota_before_seed_percent=100*w['scouts_quota_before_seed']/routes,
        recall_percent=100*x['mean_recall'],severe=x['severe_events'],events=x['graph_query_events']))
result=dict(scope='Descriptive historical development: Java COSINE, 2000 previously used T2I queries, three Lucene graphs, total actual work p+6400. The pre-seed budget event does not isolate incomplete upper descent. Not a prospective proof that 400 is optimal.',
            sources={'depth-allocation-20260910/DEVELOPMENT-RESULTS.json':hashlib.sha256(S.read_bytes()).hexdigest()},rows=rows)
(R/'R2-DEPTH-EVIDENCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
