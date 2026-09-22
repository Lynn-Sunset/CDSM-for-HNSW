"""Independent stdlib re-aggregation of downloaded server records; no ANN."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math,statistics,tarfile,struct

R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b)
def quantile(a,q):
    a=sorted(a);pos=(len(a)-1)*q;lo=int(pos)
    return a[lo]+(a[min(lo+1,len(a)-1)]-a[lo])*(pos-lo)

bundle=read(R/'SERVER-BUNDLE.json')
assert sha(R/bundle['archive'])==bundle['archive_sha256']
out=R/'server-results'
if not out.exists():
    out.mkdir()
    with tarfile.open(R/bundle['archive'],'r:gz')as t:
        assert all(p.isfile() for p in t.getmembers())
        t.extractall(out,filter='data')
assert all((out/f['path']).stat().st_size==f['bytes'] and sha(out/f['path'])==f['sha256'] for f in bundle['files'])
P=read(out/'PLAN.json');S=read(out/'SUMMARY.json');A=read(out/'AUDIT.json');SEL=read(out/'SELECTION.json')
assert sha(out/'SELECTION.json')==read(out/'SELECTION-FROZEN.json')['sha256']
assert A['all_records_spend_exact_cap'] and A['timing']['quality_signature_parity']
dev_ids=set(map(int,(out/'queries/development.txt').read_text().split()))
eval_ids=set(map(int,(out/'queries/evaluation.txt').read_text().split()))
time_ids=set(map(int,(out/'queries/timing-evaluation.txt').read_text().split()))
assert len(dev_ids)==1000 and len(eval_ids)==8000 and len(time_ids)==800 and not dev_ids&eval_ids and time_ids<=eval_ids
truth_file=out/'archive-dependencies/top10.ivecs'
assert sha(truth_file)==next(x['sha256']for x in P['inputs']if x['path']==P['data']['gt'])
truth=[]
for row in struct.iter_unpack('<11i',truth_file.read_bytes()):
    assert row[0]==10 and len(set(row[1:]))==10
    truth.append(set(row[1:]))
quality={};counts={}
for group,wanted in [('development',dev_ids),('evaluation',eval_ids)]:
    n=0
    for p in sorted((out/'results').glob('quality-'+group+'-*.csv')):
        with p.open()as f:
            for row in csv.DictReader(f):
                name=row['method'];policy=row['policy'];graph=name[:-(len(policy)+1)]
                cap=int(row['param']);qid=int(row['qid']);key=(group,graph,policy,cap)
                if key not in quality:quality[key]={}
                assert qid not in quality[key] and qid in wanted
                assert int(row['dc'])==cap
                answers=[int(x)for x in row['ids'].split(';')if int(x)>=0]
                assert len(answers)==len(set(answers)) and int(row['hits'])==len(set(answers)&truth[qid])
                quality[key][qid]=(int(row['hits']),int(row['dc']),int(row['primary']),int(row['scout']),int(row['upper_dc']),int(row['scouts']))
                n+=1
    counts[group]=n
    for key,rows in quality.items():
        if key[0]==group:assert set(rows)==wanted,key
assert counts=={'development':92000,'evaluation':736000}
for g in [x['name']for x in P['graphs']]:
    for cap in P['caps']:
        def rank(policy):
            rows=quality['development',g,policy,cap].values()
            return (-sum(x[0]for x in rows),sum(x[0]<5 for x in rows),int(policy[3:]))
        assert min(['MEP1','MEP4','MEP8','MEP16'],key=rank)==SEL['selections'][g+'-'+str(cap)]

latencies=defaultdict(lambda:defaultdict(dict));n=0
for p in sorted((out/'results').glob('timing-*.csv')):
    with p.open()as f:
        for row in csv.DictReader(f):
            policy=row['policy'];graph=row['method'][:-(len(policy)+1)]
            key=(graph,policy,int(row['param']));qid=int(row['qid']);rep=int(row['rep'])
            assert qid in time_ids and rep not in latencies[key][qid]
            latencies[key][qid][rep]=int(row['ns'])
            assert tuple(int(row[k])for k in ['hits','dc','primary','scout','upper_dc','scouts'])==quality[('evaluation',)+key][qid]
            n+=1
assert n==368000
for cell in S['cells']:
    key=(cell['graph'],cell['policy'],cell['cap']);rows=list(quality[('evaluation',)+key].values())
    close(cell['recall'],sum(r[0]for r in rows)/80000)
    assert cell['severe']==sum(r[0]<5 for r in rows) and cell['zero']==sum(r[0]==0 for r in rows)
    for i,metric in [(1,'mean_dc'),(2,'mean_primary_dc'),(3,'mean_scout_dc'),(4,'mean_upper_dc'),(5,'mean_executed_routes')]:close(cell[metric],statistics.mean(r[i]for r in rows))
    groups=latencies[key];assert set(groups)==time_ids and all(set(v)==set(range(5))for v in groups.values())
    med=[statistics.median(v.values())for v in groups.values()]
    close(cell['qps'],1e9/statistics.mean(med));close(cell['mean_ms'],statistics.mean(med)/1e6)
    close(cell['p95_ms'],quantile(med,.95)/1e6);close(cell['p99_ms'],quantile(med,.99)/1e6)
    for rep in range(5):close(cell['round_qps'][rep],1e9/statistics.mean(v[rep]for v in groups.values()))

rawp=[]
for test in S['primary_tests']:
    g=test['graph'];b=test['actual_baseline'];cap=test['cap']
    a=quality['evaluation',g,b,cap];f=quality['evaluation',g,'F',cap]
    rescue=sum(a[q][0]<5<=f[q][0]for q in eval_ids);harm=sum(f[q][0]<5<=a[q][0]for q in eval_ids)
    assert(rescue,harm)==(test['rescued'],test['harmed'])
    close(test['recall_delta_pp'],sum(f[q][0]-a[q][0]for q in eval_ids)/800)
    n=rescue+harm
    prob=1.0 if not n else min(1.,math.ldexp(sum(math.comb(n,j)for j in range(min(rescue,harm)+1)),-n+1))
    close(prob,test['mcnemar_p']);rawp.append(prob)
adjusted={};running=0.
for rank,i in enumerate(sorted(range(len(rawp)),key=lambda i:rawp[i])):
    running=max(running,min(1.,rawp[i]*(len(rawp)-rank)));adjusted[i]=running
for i,test in enumerate(S['primary_tests']):close(test['holm_p'],adjusted[i])
report=dict(passed=True,scope='Independent stdlib raw-record aggregation, selection, exact McNemar/Holm and timing quantiles; no ANN, no bootstrap interval reimplementation.',
    bundle_sha256=bundle['archive_sha256'],bundle_files_verified=len(bundle['files']),development_records=92000,evaluation_records=736000,timing_records=368000,
    cells_verified=len(S['cells']),primary_tests_verified=len(S['primary_tests']),all_actual_dc_equal=True,source_quality_timing_parity=True,ground_truth_intersections_verified=828000,
    summary_sha256=sha(out/'SUMMARY.json'),audit_sha256=sha(out/'AUDIT.json'))
(R/'LOCAL-VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
