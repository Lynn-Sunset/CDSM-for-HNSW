"""Descriptive paper analyses from frozen records; does not execute ANN search."""
from pathlib import Path
from collections import Counter
from statistics import mean
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'source-data'
sources = {}

def source(rel):
    p = ROOT / rel
    sources[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def load(rel):
    return json.loads(source(rel).read_text(encoding='utf-8-sig'))

factorial = load('t2i-scout-factorial-20260913/analysis/RESULTS.json')
arms = {p: {} for p in ['C', 'L_LOCAL']}
with source('t2i-scout-factorial-20260913/analysis/event-phases.csv').open(newline='', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if r['policy'] not in arms:
            continue
        key = (int(r['seed']), int(r['qid']))
        assert key not in arms[r['policy']]
        arms[r['policy']][key] = (int(r['final_hits']), int(r['dc']))
c, f = arms['C'], arms['L_LOCAL']
assert c.keys() == f.keys() and len(c) == 23997
assert len({qid for seed, qid in c}) == 7999
assert all(c[k][1] == f[k][1] for k in c)
histogram = {p: [sum(h == value for h, dc in rows.values()) for value in range(11)] for p, rows in arms.items()}
for p, counts in histogram.items():
    ref = factorial['summary'][p]
    assert sum(counts) == ref['events']
    assert sum(i * n for i, n in enumerate(counts)) == ref['hits']
    assert sum(counts[:5]) == ref['severe'] and counts[0] == ref['zero']
thresholds = []
for t in [1, 3, 5, 7, 9, 10]:
    thresholds.append(dict(hits_below=t, C=sum(h<t for h, dc in c.values()), F=sum(h<t for h, dc in f.values()),
        rescues=sum(c[k][0]<t<=f[k][0] for k in c), harms=sum(f[k][0]<t<=c[k][0] for k in c)))
rescue_pairs = Counter((c[k][0], f[k][0]) for k in c if c[k][0]<5<=f[k][0])

native = load('hnsw-adaptive-baselines-20260920/server-results/final-20260921-005721/results/SUMMARY.json')
fixed = load('fanng-search-comparison-20260922/SUMMARY.json')
graphs = fixed['protocol']['graphs']
points = []
for g in graphs:
    def point(p, b):
        r = next(r for r in native['evaluation'][f'{g}-{p}'] if r['parameter']==b)
        assert r['mean_dc'] == b and r['queries'] == 8000
        return r
    cr, fr = point('C',11264), point('F',10240)
    assert fr['recall']>cr['recall'] and fr['severe']<cr['severe'] and fr['qps']>cr['qps']
    points.append(dict(graph=g,C_cap=11264,F_cap=10240,C=cr,F=fr,qps_gain_pct=100*(fr['qps']/cr['qps']-1)))

tail=load('laion-abs-tail-20260922/TAIL.json')
cost=[]
for parameter in [30,50,58,60,90]:
    rows=[tail['graphs'][str(seed)][str(parameter)] for seed in [42,777,1234]]
    cost.append(dict(gamma=parameter/1000,mean_dc=mean(r['scores']['mean'] for r in rows),
        mean_ms=mean(r['timing_ms']['mean'] for r in rows),
        p99_min_ms=min(r['timing_ms']['quantiles']['0.99'] for r in rows),
        p99_max_ms=max(r['timing_ms']['quantiles']['0.99'] for r in rows)))
out=dict(scope='Retrospective descriptive analyses of existing records; no new ANN executions or new confirmatory tests.',
    sources=sources,graphs=graphs,events=23997,query_identities=7999,actual_dc_equal=True,
    histogram=histogram,thresholds=thresholds,
    severe_rescue_transitions=[dict(C_hits=a,F_hits=b,count=n) for (a,b),n in sorted(rescue_pairs.items())],
    rescues=sum(rescue_pairs.values()),rescues_4_to_5=rescue_pairs[4,5],
    rescues_to_at_least_8=sum(n for (a,b),n in rescue_pairs.items() if b>=8),
    workpoints=points,workpoint_timing='Earlier native wave: median round throughput. Post-hoc selected operating points, not fixed-cap wave timing or equal elapsed time.',
    abs_cost=cost,
    fixed_pairs=[dict(graph=r['graph'],baseline=r['baseline'],rescue=r['rescue'],harm=r['harm'],risk_change_pp=r['severe_risk_delta_pp_F_minus_baseline']) for r in fixed['contrasts'] if r['cap']==10240])
(HERE/'REVISION-DATA.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(events=out['events'],rescues=out['rescues'],rescues_to_at_least_8=out['rescues_to_at_least_8'],source_files=len(sources))))
