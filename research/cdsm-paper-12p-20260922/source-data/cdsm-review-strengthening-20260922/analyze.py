"""Frozen R2 paired analysis, source/identity audits, and development selection."""
from pathlib import Path
import csv,hashlib,json,math,statistics,struct,sys
import numpy as np

R=Path(__file__).resolve().parent;ROOT=R.parent.parent
P=json.loads((R/'PLAN.json').read_text())
FIELDS=['hits','dc','primary','scout','upper_dc','logical','edges','expansions','stop','sequence','ids','distances',
        'primary_sequence','primary_stop','phase_end','phase_sequence','scouts']
LEGACY=FIELDS[:12]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==text,('preserve artifact',path)
    else:path.write_text(text)
def ids(group):return list(map(int,(R/'queries'/f'{group}.txt').read_text().split()))
def caps(policy):return P['sensitivity_caps']if policy in P['sensitivity']else P['caps']
def policies():return P['policies']+P['sensitivity']
def signature(row):return hashlib.sha256('|'.join(row[k]for k in FIELDS).encode()).digest()
def mcnemar(rescued,harmed):
    n=rescued+harmed
    return 1.0 if not n else min(1.0,2*sum(math.comb(n,k)for k in range(min(rescued,harmed)+1))/(1<<n))
def holm(values):
    result=[0.]*len(values);last=0.
    for rank,i in enumerate(sorted(range(len(values)),key=values.__getitem__)):
        last=max(last,min(1.,values[i]*(len(values)-rank)));result[i]=last
    return result
def interval(values,denominator):
    values=np.asarray(values,dtype=np.int64);unique,counts=np.unique(values,return_counts=True)
    rng=np.random.default_rng(P['bootstrap_seed'])
    samples=rng.multinomial(len(values),counts/len(values),size=P['bootstrap_repetitions'])
    changes=samples@unique/len(values)/denominator*100.
    return np.quantile(changes,[.025,.975]).tolist()
def read(group):
    expected=ids(group);wanted=set(expected);truthdata=(ROOT/P['data']['gt']).read_bytes()
    truth=[]
    for row in struct.iter_unpack('<11i',truthdata):
        assert row[0]==10 and len(set(row[1:]))==10;truth.append(set(row[1:]))
    result={};receipts={};n=0
    for g in P['graphs']:
        for policy in policies():
            path=R/'results'/f'quality-{group}-{g["name"]}-{policy}.csv'
            raw={};receipts[str(path.relative_to(R))]=sha(path)
            with path.open()as f:
                for row in csv.DictReader(f):
                    qid=int(row['qid']);cap=int(row['param']);key=(cap,qid)
                    assert key not in raw and qid in wanted and cap in caps(policy)
                    assert row['policy']==policy and row['method']==g['name']+'-'+policy and int(row['rep'])==0
                    answers=[int(x)for x in row['ids'].split(';')if int(x)>=0]
                    assert len(answers)==len(set(answers))and int(row['hits'])==len(set(answers)&truth[qid])
                    distances=list(map(float,row['distances'].split(';')))
                    assert distances==sorted(distances) and not any(math.isnan(v)for v in distances)
                    dc=int(row['dc']);primary=int(row['primary']);scout=int(row['scout'])
                    assert 0<dc<=cap and 0<=primary<=min(cap,6400) and 0<=scout<=dc-primary
                    assert int(row['phase_end'])==primary+scout and int(row['stop'])==(0 if dc==cap else 2)
                    route=[int(x)for x in row['route_dc'].split(';')if x]
                    assert sum(route)==scout
                    assert int(row['upper_dc'])<=dc and int(row['logical'])>=dc
                    if policy in ['F','F200','F800','FRONTIER']:
                        quota={'F':400,'F200':200,'F800':800,'FRONTIER':400}[policy]
                        assert all(0<=v<=quota for v in route)and int(row['scouts'])<=4
                    if policy.startswith('MEP'):
                        assert int(row['scouts'])<=int(policy[3:])
                        upper=[int(x)for x in row['route_upper_dc'].split(';')if x]
                        assert upper==route,'MEP must have no private bottom exploration'
                    raw[key]=(int(row['hits']),dc,primary,scout,int(row['upper_dc']),signature(row),
                              row['primary_sequence'],row['primary_stop'],int(row['scouts']))
                    n+=1
            assert set(raw)=={(cap,qid)for cap in caps(policy)for qid in wanted},path
            for cap in caps(policy):result[(g['name'],policy,cap)]={qid:raw[(cap,qid)]for qid in expected}
    prefix_count=0
    for(g,policy,cap),rows in result.items():
        baseline=result[(g,'C',cap)]
        for qid,row in rows.items():
            assert row[2]==baseline[qid][2] and row[6:8]==baseline[qid][6:8],('common prefix',g,policy,cap,qid)
            prefix_count+=1
    return result,{'records':n,'source_hashes':receipts,'common_prefix_verified_records':prefix_count}
def metrics(rows):
    a=list(rows.values());n=len(a)
    return {'queries':n,'recall':sum(v[0]for v in a)/(10*n),'severe':sum(v[0]<5 for v in a),
            'zero':sum(v[0]==0 for v in a),'mean_dc':statistics.mean(v[1]for v in a),
            'mean_primary_dc':statistics.mean(v[2]for v in a),'mean_scout_dc':statistics.mean(v[3]for v in a),
            'mean_upper_dc':statistics.mean(v[4]for v in a),'mean_executed_routes':statistics.mean(v[8]for v in a)}
def select():
    data,audit=read('development');selection={};grid={}
    for g in P['graphs']:
        for cap in P['caps']:
            ms={p:metrics(data[g['name'],p,cap])for p in ['MEP1','MEP4','MEP8','MEP16']}
            chosen=min(ms,key=lambda p:(-ms[p]['recall'],ms[p]['severe'],int(p[3:])))
            selection[g['name']+'-'+str(cap)]=chosen;grid[g['name']+'-'+str(cap)]=ms
    result={'rule':P['selection'],'selections':selection,'development':grid,'audit':audit,
            'PLAN_sha256':sha(R/'PLAN.json'),'analysis_sha256':sha(R/'analyze.py')}
    save(R/'SELECTION.json',result);print('DEVELOPMENT_SELECTION_FROZEN',json.dumps(selection))
def paired(left,right):
    order=sorted(left);assert set(left)==set(right)
    a=np.array([left[q][0]for q in order],dtype=np.int64);b=np.array([right[q][0]for q in order],dtype=np.int64)
    rescue=int(((a<5)&(b>=5)).sum());harm=int(((a>=5)&(b<5)).sum())
    return {'queries':len(order),'recall_delta_pp':float((b-a).mean()*10),
            'recall_delta_ci95_pp':interval(b-a,10),'severe_delta_pp':float(((b<5).astype(int)-(a<5)).mean()*100),
            'severe_delta_ci95_pp':interval((b<5).astype(int)-(a<5).astype(int),1),
            'rescued':rescue,'harmed':harm,'improved':int((b>a).sum()),'worse':int((b<a).sum()),
            'mcnemar_p':mcnemar(rescue,harm),
            'equal_actual_dc':all(left[q][1]==right[q][1]for q in order)}
def timing(data):
    expected=ids('timing-evaluation');out={};receipts={};records=0
    for g in P['graphs']:
        for policy in policies():
            groups={cap:{qid:[]for qid in expected}for cap in caps(policy)}
            for rep in range(P['timing_rounds']):
                path=R/'results'/f'timing-{g["name"]}-{policy}-r{rep}.csv';seen=set()
                receipts[str(path.relative_to(R))]=sha(path)
                with path.open()as f:
                    for row in csv.DictReader(f):
                        qid=int(row['qid']);cap=int(row['param']);key=(cap,qid)
                        assert key not in seen and cap in groups and qid in groups[cap]and int(row['rep'])==rep
                        assert row['policy']==policy and row['method']==g['name']+'-'+policy
                        assert signature(row)==data[g['name'],policy,cap][qid][5],('timing quality parity',path,qid,cap)
                        ns=int(row['ns']);assert ns>0;groups[cap][qid].append(ns);seen.add(key);records+=1
                assert seen=={(cap,qid)for cap in caps(policy)for qid in expected}
            for cap,rows in groups.items():
                mat=np.array([rows[qid]for qid in expected],dtype=np.float64)
                med=np.median(mat,axis=1);mean=float(med.mean());rounds=(1e9/mat.mean(axis=0)).tolist()
                out[(g['name'],policy,cap)]={'qps':1e9/mean,'mean_ms':mean/1e6,
                    'p95_ms':float(np.quantile(med,.95)/1e6),'p99_ms':float(np.quantile(med,.99)/1e6),
                    'round_qps':rounds,'timing_queries':len(expected),'rounds':P['timing_rounds']}
    return out,{'records':records,'source_hashes':receipts,'quality_signature_parity':True}
def final():
    data,audit=read('evaluation');dev=json.loads((R/'SELECTION.json').read_text());times,timeaudit=timing(data)
    cells=[];bykey={};underfilled=0
    for key,rows in data.items():
        g,p,b=key;m=metrics(rows);m.update(times[key]);m.update(graph=g,policy=p,cap=b)
        m['underfilled']=sum(v[1]!=b for v in rows.values());underfilled+=m['underfilled'];cells.append(m);bykey[key]=m
    contrasts=[]
    for g in P['graphs']:
        cap=P['primary_cap'];chosen=dev['selections'][g['name']+'-'+str(cap)]
        for label,baseline in [('MEP4','MEP4'),('MEP_DEV',chosen),('FRONTIER','FRONTIER')]:
            c=paired(data[g['name'],baseline,cap],data[g['name'],'F',cap]);c.update(graph=g['name'],cap=cap,baseline=label,actual_baseline=baseline)
            c['qps_delta_percent']=100*(times[g['name'],'F',cap]['qps']/times[g['name'],baseline,cap]['qps']-1)
            contrasts.append(c)
    for c,p in zip(contrasts,holm([c['mcnemar_p']for c in contrasts])):c['holm_p']=p
    descriptive=[]
    for g in P['graphs']:
        for cap in P['caps']:
            for baseline in ['C','MEP1','MEP4','MEP8','MEP16','FRONTIER']:
                c=paired(data[g['name'],baseline,cap],data[g['name'],'F',cap]);c.update(graph=g['name'],cap=cap,baseline=baseline,policy='F')
                c['qps_delta_percent']=100*(times[g['name'],'F',cap]['qps']/times[g['name'],baseline,cap]['qps']-1);descriptive.append(c)
        for policy in P['sensitivity']:
            c=paired(data[g['name'],'C',10240],data[g['name'],policy,10240]);c.update(graph=g['name'],cap=10240,baseline='C',policy=policy)
            c['qps_delta_percent']=100*(times[g['name'],policy,10240]['qps']/times[g['name'],'C',10240]['qps']-1);descriptive.append(c)
    clusters=[];order=ids('evaluation')
    for label in ['C','MEP4','MEP_DEV','FRONTIER']:
        dif=[];risks=[]
        for q in order:
            ds=[];rs=[]
            for g in P['graphs']:
                baseline=dev['selections'][g['name']+'-10240']if label=='MEP_DEV'else label
                a=data[g['name'],baseline,10240][q][0];b=data[g['name'],'F',10240][q][0]
                ds.append(b-a);rs.append(int(b<5)-int(a<5))
            dif.append(sum(ds));risks.append(sum(rs))
        clusters.append({'baseline':label,'query_clusters':len(order),'fixed_graphs':4,
            'recall_delta_pp':statistics.mean(dif)/40*100,'recall_delta_ci95_pp':interval(dif,40),
            'severe_delta_pp':statistics.mean(risks)/4*100,'severe_delta_ci95_pp':interval(risks,4),
            'scope':'Descriptive fixed-graph query-cluster intervals; not multiplicity-adjusted or independent fresh-query confirmation.'})
    audit.update(underfilled_records=underfilled,all_records_spend_exact_cap=underfilled==0,
                 timing=timeaudit,legacy=json.loads((R/'engineering/LEGACY-PARITY.json').read_text()),
                 expected_quality_records=P['expected_evaluation_records'],expected_timing_records=P['expected_timing_records'])
    assert audit['records']==P['expected_evaluation_records']and timeaudit['records']==P['expected_timing_records']
    summary={'scope':P['scope'],'cells':cells,'primary_tests':contrasts,'descriptive_contrasts':descriptive,
             'query_cluster_descriptive':clusters,'selection':dev['selections'],
             'statistical_scope':P['primary_tests'],'QPS_definition':'Inverse mean of per-query median latency over five rounds; serial one-thread queries; 1.5% practical tolerance, not statistical equivalence.',
             'historical_query_exposure':True,'paired_intervals':'5000 paired query bootstrap draws; per-comparison descriptive 95% intervals, not simultaneous.',
             'all_actual_dc_equal':underfilled==0}
    save(R/'SUMMARY.json',summary);save(R/'AUDIT.json',audit)
    lines=['# CDSM R2：服务器最近对照与深度补强结果','',
           '全部ANN运行与正式时间来自服务器。本轮策略和分析在开发前冻结；评价查询已在历史实验中暴露。',
           f'质量记录：{audit["records"]:,}；计时记录：{timeaudit["records"]:,}；未耗满总预算：{underfilled}。',
           '','## 主预算10,240','',
           '|图|方法|Recall %|严重失败/8000|零召回|QPS|p99查询中位延迟 ms|平均备用DC|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in P['graphs']:
        for p in policies():
            m=bykey[g['name'],p,10240]
            lines.append(f'|{g["name"]}|{p}|{100*m["recall"]:.5f}|{m["severe"]}|{m["zero"]}|{m["qps"]:.2f}|{m["p99_ms"]:.3f}|{m["mean_scout_dc"]:.2f}|')
    lines+=['','## 预定主要比较：后者为F','',
            '|图|对照（实际策略）|召回差 pp|救援/伤害|Holm p|F的QPS差 %|','|---|---|---:|---:|---:|---:|']
    for c in contrasts:lines.append(f'|{c["graph"]}|{c["baseline"]} ({c["actual_baseline"]})|{c["recall_delta_pp"]:+.5f}|{c["rescued"]}/{c["harmed"]}|{c["holm_p"]:.6g}|{c["qps_delta_percent"]:+.3f}|')
    lines+=['','全部参数与预算、开发选择及逐查询审计见SUMMARY.json、SELECTION.json、AUDIT.json。MEP-DEV与MEP4重合时不构成两份独立证据。FRONTIER是自定义的当前前沿局部探索控制，不是原生iQAN。',
            '','QPS在本轮内部比较；p99是每查询重复中位数的分位数，不是线上服务尾延迟。小于等于1.5%的点差仅按实用容差解释。描述性bootstrap区间不承担多重比较校正后的主检验。']
    path=R/'REPORT.zh-CN.md';content='\n'.join(lines)+'\n'
    if path.exists():assert path.read_text()==content
    else:path.write_text(content)
    print(json.dumps({'analysis_complete':True,'evaluation_records':audit['records'],'timing_records':timeaudit['records'],
                      'underfilled':underfilled,'primary_tests':contrasts},ensure_ascii=False))
def selftest():
    assert mcnemar(0,0)==1 and mcnemar(8,0)==2/256 and mcnemar(3,3)==1
    assert holm([.01,.04,.03])==[.03,.06,.06]
    assert interval([1]*20,10)==[10.,10.]
    assert interval([0]*20,1)==[0.,0.]
    print('R2_ANALYSIS_SELFTEST_PASS')
if __name__=='__main__':{'select':select,'final':final,'selftest':selftest}[sys.argv[1]]()
