"""Frozen full-factorial descriptive analysis, with explicit retrospective scope."""
from pathlib import Path
import collections
import csv
import json
import numpy as np

H=Path(__file__).resolve().parent
ARMS=['C','L_LOCAL','L_GLOBAL','L_NONE','A_LOCAL','A_GLOBAL','A_NONE']
PAIRS=[(f'A_{s}',f'L_{s}') for s in ['LOCAL','GLOBAL','NONE']]+[
    (f'{a}_{s1}',f'{a}_{s0}') for a in ['L','A'] for s1,s0 in [('LOCAL','GLOBAL'),('NONE','GLOBAL'),('NONE','LOCAL')]]+[
    ('L_LOCAL','A_NONE')]+[(a,'C') for a in ARMS[1:]]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def ints(s):return [int(v) for v in s.split(';') if v!='']
def csvwrite(p,data):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
def gt(p):
    with Path(p).open('rb') as f:
        assert list(np.fromfile(f,dtype='>i4',count=2))==[7999,50]
        data=np.fromfile(f,dtype=np.dtype([('id','>i4'),('score','>f4')]))
    assert data.size==7999*50 and np.isfinite(data['score']).all()
    return [set(map(int,a)) for a in data['id'].reshape(7999,50)[:,:10]]
def stats(data):
    n=len(data);assert n
    return dict(events=n,queries=len({r['qid'] for r in data}),hits=sum(r['h'] for r in data),
        recall_percent=sum(r['h'] for r in data)/n*10,severe=sum(r['h']<5 for r in data),zero=sum(r['h']==0 for r in data),
        **{f'mean_{k}':sum(float(r[k]) for r in data)/n for k in ['dc','logical','edges','expansions']})
def interval(qid_differences,rng):
    values,counts=np.unique(list(qid_differences.values()),return_counts=True);n=len(qid_differences)
    draws=(rng.multinomial(n,counts/counts.sum(),size=20000)@values)/(n*3)*100
    return list(map(float,np.quantile(draws,[.025,.975],method='linear')))
def compare(a,b,rng):
    assert a.keys()==b.keys();c=collections.Counter();d={};phase=collections.Counter()
    for key,r in a.items():
        ref=b[key];difference=int(ref['h']<5)-int(r['h']<5);d[key[1]]=d.get(key[1],0)+difference
        c['rescues']+=ref['h']<5<=r['h'];c['harms']+=r['h']<5<=ref['h'];c['wins']+=r['h']>ref['h'];c['losses']+=r['h']<ref['h'];c['net_hits']+=r['h']-ref['h']
        c['same_returned_set']+=set(ints(r['hits']))==set(ints(ref['hits']))
        if ref['h']<5<=r['h']:
            phase['rescues_target_already_nonsevere_post_scout']+=r['post_h']>=5
            phase['rescues_target_crosses_threshold_in_continuation']+=r['post_h']<5
    assert len(a)==3*len(d)
    return dict(events=len(a),query_clusters=len(d),**dict(c),ties=len(a)-c['wins']-c['losses'],
        severe_reduction_pp=sum(d.values())/len(a)*100,descriptive_ci95_pp=interval(d,rng),rescue_stage=dict(phase),
        mean_dc_ratio=sum(int(r['dc']) for r in a.values())/sum(int(r['dc']) for r in b.values()))
def selftest():
    a={};b={}
    for seed in [42,777,1234]:
        key=(seed,7);base=dict(qid=7,h=4,post_h=3,hits='1;2',dc=10,logical=11,edges=20,expansions=2)
        b[key]=base;a[key]=dict(base,h=5,post_h=4)
    r=compare(a,b,np.random.default_rng(1));assert r['rescues']==3 and r['harms']==0 and r['severe_reduction_pp']==100 and r['descriptive_ci95_pp']==[100.,100.]
    assert r['rescue_stage']['rescues_target_crosses_threshold_in_continuation']==3
    assert stats(list(a.values()))['recall_percent']==50
    return dict(status='passed',checks=['paired severe threshold and signed difference','query cluster interval','phase crossing','recall denominator'])
def table(headers,data):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in data])

def main():
    out=H/'analysis';out.mkdir(exist_ok=False);write(out/'ANALYSIS-SELFTEST.json',selftest());plan=read(H/'PLAN.json')
    byarm={a:{} for a in ARMS};original_s=[];regressions=collections.Counter();allrows=[]
    for g in plan['graphs']:
        seed=g['seed'];truth=gt(g['gt_file']);old={(int(r['qid']),r['policy']):r for r in rows(g['effect_csv'])}
        restart={int(r['qid']):r for r in rows(g['priority_effect_csv']) if r['policy']=='R_LOCAL'}
        new=rows(H/'effects'/f's{seed}'/'effect.csv');assert len(new)==7999*len(ARMS)
        for r in new:
            pos=int(r['position']);qid=int(r['qid']);arm=r['policy'];key=(seed,qid);assert key not in byarm[arm] and pos==int(old[qid,'C']['position'])
            answer=ints(r['hits']);assert len(set(answer))==len(answer)==10
            r.update(seed=seed,qid=qid,position=pos,h=len(set(answer)&truth[pos]),primary_h=len(set(ints(r['primary_hits']))&truth[pos]),post_h=len(set(ints(r['post_scout_hits']))&truth[pos]))
            p=int(r['primary_dc']);dc=int(r['dc']);assert dc<=p+6400<=12800
            assert p<=int(r['c_lcp'])<=dc and p<=int(r['f_lcp'])<=dc
            for prefix in ['c','f']:
                assert p<=int(r[prefix+'_intersection'])<=dc and 0<=float(r[prefix+'_jaccard'])<=1 and 0<=float(r[prefix+'_post_jaccard'])<=1
            costs=ints(r['route_dc']);upper=ints(r['route_upper_dc']);reasons=ints(r['route_reason']);assert len(costs)==len(upper)==len(reasons)==4
            assert all(0<=u<=c<=400 for u,c in zip(upper,costs))
            assert sum(costs)==int(r['scout_dc']) and p+sum(costs)==int(r['after_scout_dc'])<=dc
            r['route_h']=[len(set(ints(s))&truth[pos]) for s in r['route_hits'].split('|')]
            assert len(r['route_h'])==4
            if arm!='C':assert r['route_h'][-1]==r['post_h']
            if arm in ['C','L_LOCAL','A_GLOBAL']:
                ref=restart[qid] if arm=='A_GLOBAL' else old[qid,'C' if arm=='C' else 'F']
                for k in ['position','hits','dc','logical','edges','expansions','primary_dc','primary_stop']:assert str(r[k])==ref[k],(seed,qid,arm,k)
                if arm=='A_GLOBAL':
                    for k in ['sequence','primary_sequence','entries','eps','route_dc','route_upper_dc']:assert r[k]==ref[k],(seed,qid,k)
                regressions[arm]+=1
            byarm[arm][key]=r;allrows.append(r)
        for (qid,arm),r in old.items():
            if arm=='S':original_s.append(dict(r,seed=seed,qid=qid,h=len(set(ints(r['hits']))&truth[int(r['position'])])))
    assert all(len(v)==23997 for v in byarm.values()) and all(v==23997 for v in regressions.values())
    for key,c in byarm['C'].items():
        for arm in ARMS:
            assert byarm[arm][key]['primary_hits']==c['primary_hits'] and byarm[arm][key]['primary_sequence']==c['primary_sequence']
        first={ints(byarm[arm][key]['entries'])[0] for arm in ARMS[1:]};assert len(first)==1
    rng=np.random.Generator(np.random.PCG64(2026091302))
    summaries={a:stats(list(v.values())) for a,v in byarm.items()};summaries['S_reference']=stats(original_s)
    comparisons={a+'/'+b:compare(byarm[a],byarm[b],rng) for a,b in PAIRS}
    interactions={}
    for end in ['LOCAL','NONE']:
        d={}
        for key in byarm['C']:
            h=lambda a:int(byarm[a][key]['h']<5)
            value=(h('L_GLOBAL')-h('L_'+end))-(h('A_GLOBAL')-h('A_'+end));d[key[1]]=d.get(key[1],0)+value
        interactions[end+'_vs_GLOBAL']=dict(L_minus_A_severe_reduction_pp=sum(d.values())/23997*100,descriptive_ci95_pp=interval(d,rng))
    work={};paths={};phase={}
    for arm,group in byarm.items():
        rr=list(group.values());costs=[];upper=[];expansions=[];reasons=[]
        for r in rr:
            for c,u,e,reason,entry in zip(ints(r['route_dc']),ints(r['route_upper_dc']),ints(r['route_expansions']),ints(r['route_reason']),ints(r['entries'])):
                if entry<0:continue
                costs.append(c);upper.append(u);expansions.append(e);reasons.append(reason)
        work[arm]=dict(mean_scout_dc=float(np.mean([int(r['scout_dc']) for r in rr])),
            mean_scout_upper_dc=sum(upper)/len(rr),mean_scout_additional_base_dc=(sum(costs)-sum(upper))/len(rr),
            mean_scout_completed_expansions=sum(expansions)/len(rr),started_routes=len(costs),
            routes_at_400=sum(c==400 for c in costs),routes_without_additional_base_scores=sum(c==u for c,u in zip(costs,upper)),
            routes_without_completed_base_expansions=sum(e==0 for e in expansions),stop_reasons=dict(collections.Counter(map(str,reasons))),
            events_actual_dc_equal_C=sum(int(r['dc'])==int(byarm['C'][key]['dc']) for key,r in group.items()))
        phase[arm]=dict(primary_hits=sum(r['primary_h'] for r in rr),post_scout_hits=sum(r['post_h'] for r in rr),
            final_hits=sum(r['h'] for r in rr),post_scout_severe=sum(r['post_h']<5 for r in rr),
            continuation_crosses_severe_threshold=sum(r['post_h']<5<=r['h'] for r in rr),
            final_minus_post_scout_hits=sum(r['h']-r['post_h'] for r in rr),
            route_hit_totals=[sum(r['route_h'][i] for r in rr) for i in range(4)] if arm!='C' else None)
        paths[arm]={}
        for prefix,ref in [('c','C'),('f','L_LOCAL')]:
            paths[arm][ref]=dict(mean_jaccard=float(np.mean([float(r[prefix+'_jaccard']) for r in rr])),
                mean_post_primary_jaccard=float(np.mean([float(r[prefix+'_post_jaccard']) for r in rr])),
                full_physical_sequence_equal=sum(int(r[prefix+'_lcp'])==int(r['dc'])==int(byarm[ref][key]['dc']) for key,r in group.items()),
                returned_set_equal=sum(set(ints(r['hits']))==set(ints(byarm[ref][key]['hits'])) for key,r in group.items()))
    bygraph={str(seed):{a:stats([r for key,r in v.items() if key[0]==seed]) for a,v in byarm.items()} for seed in [42,777,1234]}
    output=dict(status='passed',new_rows=len(allrows),retrospective=True,independent_confirmation=False,formal_timing=False,
        summary=summaries,comparisons=comparisons,interactions=interactions,route_work=work,phase_quality=phase,paths=paths,by_graph=bygraph,
        regression_events=dict(regressions),stop_reason_labels={'0':'quota','1':'local threshold','2':'global threshold','3':'active empty with deferred','4':'exhausted','5':'no unused entry'},
        intervals='descriptive paired qid-cluster percentile;20000 PCG64 draws;no multiplicity-adjusted winner selection')
    write(out/'RESULTS.json',output);csvwrite(out/'quality.csv',[dict(policy=a,**s) for a,s in summaries.items()]);csvwrite(out/'comparisons.csv',[dict(comparison=k,**v) for k,v in comparisons.items()])
    csvwrite(out/'event-phases.csv',[dict(seed=r['seed'],qid=r['qid'],position=r['position'],policy=r['policy'],primary_hits=r['primary_h'],post_scout_hits=r['post_h'],final_hits=r['h'],dc=r['dc'],scout_dc=r['scout_dc']) for r in allrows])
    report=['# 接纳 × 停止条件：完整消融结果\n',
        '本轮七臂、7999查询×三图，167979条新效果。L_LOCAL=原F；A_GLOBAL=上一轮R_LOCAL；三个锚点逐事件回归通过。实验定义与统计在新处理运行前冻结，人口已经暴露，属于回顾性机制扩展。\n',
        '## 效果与实际成本\n',
        table(['策略','Recall@10 %','严重(<0.5)','零命中','平均实际评分','平均读边'],[[a,f"{s['recall_percent']:.5f}",s['severe'],s['zero'],f"{s['mean_dc']:.2f}",f"{s['mean_edges']:.2f}"] for a,s in summaries.items()]),
        '\nL/A分别表示局部严格接纳/全部接纳；LOCAL/GLOBAL/NONE分别表示局部门槛/全局门槛/无距离提前停止。S_reference只复用旧结果，未修改或重新选择S。\n',
        '## 预定配对比较\n',
        table(['目标 / 参照','救回 / 伤害','普通命中胜 / 负','净命中','严重减少pp [描述性95%区间]'],[[k,f"{v['rescues']}/{v['harms']}",f"{v['wins']}/{v['losses']}",v['net_hits'],f"{v['severe_reduction_pp']:+.5f} [{v['descriptive_ci95_pp'][0]:+.5f}, {v['descriptive_ci95_pp'][1]:+.5f}]"] for k,v in comparisons.items()]),
        '\n正值代表目标策略较好。三图按qid成簇；区间是多项描述性结果，不代表多重比较后的确证性胜出。\n',
        '接纳×停止交互（L下的停止变更收益减去A下的相同变更收益）：\n',
        table(['停止变更','严重减少差pp','描述性95%区间'],[[k,v['L_minus_A_severe_reduction_pp'],v['descriptive_ci95_pp']] for k,v in interactions.items()]),
        '\n## Scout实际推进\n',
        table(['策略','平均scout评分','其中上层','额外基图评分','平均完成基图展开','400额度用满分路','无额外基图评分分路','总评分与C相同事件'],[[a,f"{v['mean_scout_dc']:.2f}",f"{v['mean_scout_upper_dc']:.2f}",f"{v['mean_scout_additional_base_dc']:.2f}",f"{v['mean_scout_completed_expansions']:.2f}",v['routes_at_400'],v['routes_without_additional_base_scores'],v['events_actual_dc_equal_C']] for a,v in work.items()]),
        '\n总cap为p+6400，每路上限400包含上层下降；未用额度留给最后续搜。实际scout评分不同是接纳/停止干预的中介，不是固定相同scout工作量下的直接效应。零新增物理评分不等于零展开。终止原因详见RESULTS.json。\n',
        '## 分阶段结果\n',
        table(['策略','scout后命中总数','最终命中总数','scout后严重','续搜中跨过严重阈值'],[[a,v['post_scout_hits'],v['final_hits'],v['post_scout_severe'],v['continuation_crosses_severe_threshold']] for a,v in phase.items()]),
        '\n每路后Top10由私有观察收集器重建，不修改活跃搜索状态；固定案例再与完整物理前缀重算一致。阶段相同不代表已花费相同工作，尤其C的scout后就是primary。\n',
        '## 实际路径\n',
        table(['策略','对F完整评分序列相同','对F返回集合相同','对F primary后Jaccard','对C primary后Jaccard'],[[a,v['L_LOCAL']['full_physical_sequence_equal'],v['L_LOCAL']['returned_set_equal'],f"{v['L_LOCAL']['mean_post_primary_jaccard']:.6f}",f"{v['C']['mean_post_primary_jaccard']:.6f}"] for a,v in paths.items()]),
        '\n首个入口和整个primary一致；后续访问状态可改变下一路实际入口。原始CSV保存入口/落点、分路评分/展开/接纳/延后/停止原因与阶段答案。\n',
        '## 完整性与范围\n',
        'C、L_LOCAL、A_GLOBAL各23997个事件与封存C/F/R_LOCAL效果及工作量对账；A_GLOBAL另核对旧序列及分路成本/入口。全体新路线的primary完整物理前缀相同。合成和固定真实案例覆盖完整路径、同分值、低分桥、额度中断与诊断开关。\n',
        '本轮没有新正式计时、新数据独立确认、新选择器部署或完整NSW系统复现。按既定六格结束，未依结果追加额度/入口/阈值搜索。总收尾以COMPLETION.json及独立计算路径审计为准。\n',
        '[协议](PROTOCOL.md)；[结果明细](analysis/RESULTS.json)；[前一轮解读](../literature-priority-experiments-20260913/INTERPRETATION.md)。\n']
    with (H/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report))
    write(out/'QA.json',dict(status='passed',new_rows=len(allrows),arms=7,events_per_arm=23997,regressions=dict(regressions),
        all_new_primary_prefixes_equal=True,first_entry_identity_equal=True,formal_timing=False,retrospective=True,predefined_comparisons=len(comparisons)))
    print(json.dumps(dict(stage='analysis_complete',rows=len(allrows),severe={k:v['severe'] for k,v in summaries.items()})),flush=True)

if __name__=='__main__':main()
