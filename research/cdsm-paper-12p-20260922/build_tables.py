"""Generate paper tables only from completed, frozen summaries."""
from pathlib import Path
from statistics import mean
import hashlib
import json
import runpy

HERE = Path(__file__).resolve().parent
RESEARCH = HERE / 'source-data' if (HERE / 'source-data').is_dir() else HERE.parent
OUT = HERE / 'paper/tables'
runpy.run_path(str(HERE/'derive_revision_data.py'))
runpy.run_path(str(HERE/'derive_r2_depth.py'))
runpy.run_path(str(HERE/'derive_r2_native_workpoints.py'))
revision = json.loads((HERE/'REVISION-DATA.json').read_text(encoding='utf-8'))
sources = {}
def load(rel):
    p = RESEARCH / rel
    sources[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8'))

fixed = load('fanng-search-comparison-20260922/SUMMARY.json')
strength = load('cdsm-review-strengthening-20260922/SUMMARY.json')
runpy.run_path(str(HERE/'build_r2_values.py'))
family = load('cdsm-family-comparison-20260922/SUMMARY.json')
native = load('hnsw-adaptive-baselines-20260920/server-results/final-20260921-005721/results/SUMMARY.json')
transfer = {(d,k): load(f'cdsm-crossmodal-supplement-20260922/{d}/{k}/SUMMARY.json') for d in ['laion','webvid'] for k in ['absolute','family']}
graphs = fixed['protocol']['graphs']
short = dict(zip(graphs, ['Faiss M32', 'Imported 42', 'Imported 777', 'Imported 1234']))
policies = ['C','FANNG','F','SCORE','DIVERSE','KEEP','E64_RAW']
def escape(value):
    return str(value).replace('_',r'\_')
def write(name, caption, label, cols, header, rows, wide=True):
    env = 'table*' if wide else 'table'
    lines = [f'\\begin{{{env}}}[t]',r'\centering',f'\\caption{{{caption}}}',f'\\label{{{label}}}',r'\small',f'\\begin{{tabular}}{{{cols}}}',r'\toprule',
             ' & '.join(header)+r' \\',r'\midrule']
    lines += [' & '.join(str(v) for v in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}',f'\\end{{{env}}}','']
    (OUT/f'{name}.tex').write_text('\n'.join(lines),encoding='utf-8')

write('cohorts','Query cohorts. Counts are distinct identities per dataset, repeated on the indicated graphs. Timing uses subsets of these queries in separate runs.','tab:cohorts','@{}lrrl@{}',
      ['Experiment','Queries','Graphs','Prior exposure'],[
          ['T2I development','1,000','4','Historical; disjoint'],
          ['T2I server','8,000','4','Historical evaluation'],
          ['Earlier F/C','7,999','3','Unused at validation'],
          ['MEP comparison','1,600','3','Newly admitted'],
          ['LAION transfer','8,000','3','Historical'],
          ['WebVid transfer','8,000','3','Historical'],
      ],wide=False)

m = {(r['graph'],r['policy'],r['cap']):r for r in strength['cells']}
rows=[]
for p in ['C','F','MEP1','MEP4','MEP8','MEP16','FRONTIER','F200','F800']:
    label={'F':'F (400)','F200':'F (200)','F800':'F (800)'}.get(p,p.replace('MEP','MEP-'))
    rows.append([label]+[f"{m[g,p,10240]['recall']*100:.4f} / {m[g,p,10240]['severe']} / {m[g,p,10240]['qps']:.1f}" for g in graphs])
write('fixed',r'T2I at exactly 10,240 evaluations per query. Each cell is mean Recall@10 (\%) / severe failures out of 8,000 / QPS. All times are remeasured together in the latest control wave. F (200/800) changes only the per-scout limit; F (400) is the frozen original.','tab:fixed','@{}lrrrr@{}',
      ['Policy']+[short[g] for g in graphs],rows)

def pvalue(v):
    if v>=.001:return f'{v:.3f}'
    mantissa,exponent=f'{v:.2e}'.split('e')
    return rf'${mantissa}\!\times\!10^{{{int(exponent)}}}$'
tests={(r['graph'],r['baseline']):r for r in strength['primary_tests']}
rows=[]
for g in graphs:
    row=[short[g],strength['selection'][g+'-10240'].replace('MEP','')]
    for control in ['MEP4','MEP_DEV','FRONTIER']:
        r=tests[g,control]
        row.append(f"{r['rescued']}/{r['harmed']} ({pvalue(r['holm_p'])})")
    rows.append(row)
write('closest-tests',r'Primary severe-failure comparisons at 10,240. Entries are rescues/harms when replacing the control with F, followed by the exact McNemar $p$ value after Holm correction across twelve tests. DEV gives the development-selected MEP entrance count; duplicated MEP choices are not independent evidence.','tab:closest','@{}lrrrr@{}',
      ['Graph',r'DEV $m$','MEP-4 to F','MEP-DEV to F','FRONTIER to F'],rows)

vm = {(r['graph'],r['policy'],r['extra']):r for r in family['metrics']}
rows=[]
for p in policies:
    r=[vm[g,p,3200] for g in graphs]
    rd=[100*(vm[g,p,3200]['recall']-vm[g,'C',3200]['recall']) for g in graphs]
    speed=[100*(vm[g,p,3200]['qps']/vm[g,'C',3200]['qps']-1) for g in graphs]
    rows.append([escape(p),' / '.join(str(x['severe']) for x in r),' / '.join(str(x['zero']) for x in r),
                 f'{min(rd):+.3f} to {max(rd):+.3f}',f'{min(speed):+.2f} to {max(speed):+.2f}'])
write('family',r'T2I family at $E=3{,}200$. Graph order: Faiss M32, imported 42, 777, 1234; each count is out of 8,000. Ranges are per-graph differences from C, not pooled estimates.','tab:family','@{}lrrrr@{}',
      ['Method','Severe counts','Zero counts','Recall change (pp)',r'QPS change (\%)'],rows)

rows=[]
for g in graphs:
    selected=[]
    for p in ['NATIVE','F','ABS','ABS_CAP']:
        key=f'{g}-{p}'
        if p=='ABS':
            dev=sorted([r for r in native['development'][key] if r['parameter']<60 and r['mean_ms']<=4],key=lambda r:(-r['recall'],r['mean_ms']))[0]
        else: dev=native['selections'][key]['latency']['4']
        r=next(x for x in native['evaluation'][key] if x['parameter']==dev['parameter'])
        selected.append(r)
    rows.append([short[g],r'R (\%) / S']+[f"{r['recall']*100:.5f} / {r['severe']}" for r in selected])
    rows.append(['','QPS / p99 ms']+[f"{r['qps']:.2f} / {r['p99_ms']:.3f}" for r in selected])
write('native',r'Four-graph points selected by development mean time $\leq4$\,ms. R is mean recall; S is severe events / 8,000. Uncapped ABS is restricted retrospectively to $\gamma<0.06$. All values belong to the earlier native wave, without equal-work or equal-time claims.','tab:native','@{}llrrrr@{}',
      ['Graph','Metric','Native','F','ABS',r'ABS\_CAP'],rows)

rows=[]
for d in ['laion','webvid']:
    j=transfer[d,'absolute']
    for p in ['C','F','FANNG']:
        r=[next(x for x in j['metrics'] if x['graph']==g and x['policy']==p and x['cap']==10240) for g in j['protocol']['graphs']]
        rows.append([d.upper(),p,f"{mean(x['recall'] for x in r)*100:.4f}",' / '.join(str(x['severe']) for x in r),
                     f"{mean(x['mean_ms'] for x in r):.3f}",f"{min(x['p99_ms'] for x in r):.3f}--{max(x['p99_ms'] for x in r):.3f}"])
write('transfer','Frozen transfer at 10,240 evaluations. Recall and mean time are equal-weight graph averages; severe counts retain seeds 42/777/1234 separately. The p99 range spans graph-level p99 values, not pooled query latencies.','tab:transfer','@{}llrrrr@{}',
      ['Dataset','Method',r'Recall (\%)','Severe / 8,000','Mean (ms)','p99 range (ms)'],rows)

rows=[]
for p in policies:
    r=[vm[g,p,6400] for g in graphs]
    row=[escape(p),f"{mean(x['recall'] for x in r)*100:.4f}",' / '.join(str(x['severe']) for x in r)]
    for d in ['laion','webvid']:
        j=transfer[d,'family']
        r=[next(x for x in j['metrics'] if x['graph']==g and x['policy']==p and x['extra']==6400) for g in j['protocol']['graphs']]
        row += [f"{mean(x['recall'] for x in r)*100:.4f}",' / '.join(str(x['severe']) for x in r)]
    rows.append(row)
write('secondary',r'Secondary family allowance $E=6{,}400$ on all datasets. Mean recall (R) is a graph average in percent; severe counts (S) remain per graph, each out of 8,000. These are descriptive secondary endpoints.','tab:secondary','@{}lrrrrrr@{}',
      ['Method','T2I R','T2I S','LAION R','LAION S','WebVid R','WebVid S'],rows)

rows=[]
for r in revision['workpoints']:
    c,f=r['C'],r['F']
    rows.append([short[r['graph']],f"{c['recall']*100:.5f} / {f['recall']*100:.5f}",f"{c['severe']} / {f['severe']}",f"{c['qps']:.2f} / {f['qps']:.2f}",f"+{r['qps_gain_pct']:.2f}\\%"])
write('more-continuation',r'Allowing C 10\% more work: C at 11,264 versus F at 10,240 evaluations, each on 8,000 queries. Both points come from the earlier native timing wave. These descriptive points were identified retrospectively in its measured grid.','tab:more','@{}lrrrr@{}',
      ['Graph',r'Recall C / F (\%)','Severe C / F','QPS C / F','F QPS gain'],rows)

factorial=load('t2i-scout-factorial-20260913/analysis/RESULTS.json')
rows=[]
for policy,label in [('C','C'),('L_GLOBAL',r'L\_GLOBAL'),('L_LOCAL',r'L\_LOCAL (F)'),('L_NONE',r'L\_NONE')]:
    r=factorial['summary'][policy];w=factorial['route_work'][policy]
    rows.append([label,f"{r['recall_percent']:.5f}",r['severe'],r['zero'],f"{w['mean_scout_additional_base_dc']:.2f}"])
write('local-stop',r'Scout stopping rules on 7,999 queries and three graphs (23,997 events). All policies match actual work per event. Bottom DC is mean new bottom-layer scoring in the scout phase; it excludes upper descent.','tab:stop','@{}lrrrr@{}',
      ['Policy',r'Recall (\%)','Severe','Zero','Bottom DC'],rows,wide=False)

write('thresholds',r'Recall-tail sensitivity of C versus F on the 7,999-query cohort (23,997 graph--query events). Counts are descriptive; threshold 5 is the original severe endpoint.','tab:thresholds','@{}rrrr@{}',
      ['Hits below','C','F','Rescue / harm'],[[r['hits_below'],r['C'],r['F'],f"{r['rescues']} / {r['harms']}"] for r in revision['thresholds']],wide=False)

tail=load('laion-abs-tail-20260922/TAIL.json')
write('abs-cost',r'LAION ABS cost context. Means average three graphs; p99 ranges span their 800-query timing subsets. Rows $\gamma\geq0.06$ are excluded from current ranking.','tab:abscost','@{}rrrr@{}',
      [r'$\gamma$','Mean DC','Mean ms','p99 ms'],[[f"{r['gamma']:.3f}",f"{r['mean_dc']:,.0f}",f"{r['mean_ms']:.3f}",f"{r['p99_min_ms']:.1f}--{r['p99_max_ms']:.1f}"] for r in revision['abs_cost']],wide=False)

rows=[]
for p in policies:
    row=[escape(p)]
    for d in ['laion','webvid']:
        j=transfer[d,'family'];rr=[next(x for x in j['metrics'] if x['graph']==g and x['policy']==p and x['extra']==3200) for g in j['protocol']['graphs']]
        row += [f"{mean(x['recall'] for x in rr)*100:.4f}",' / '.join(str(x['severe']) for x in rr)]
    rows.append(row)
write('transfer-family',r'Primary transfer allowance $E=3{,}200$: graph-average recall R (\%) and per-graph severe counts S / 8,000, in seed order 42/777/1234.','tab:transferfamily','@{}lrrrr@{}',
      ['Method','LAION R','LAION S','WebVid R','WebVid S'],rows)

load('depth-allocation-20260910/DEVELOPMENT-RESULTS.json')
depth=json.loads((HERE/'R2-DEPTH-EVIDENCE.json').read_text(encoding='utf-8'))
write('depth-history',r'Historical depth development: 2,000 T2I queries, three Lucene graphs, Java COSINE, $p+6{,}400$ actual work. $m/s$: maximum routes / per-route limit. $U$: routes exhausting their limit before bottom seeding (\%); S: severe events / 6,000. These are separate from the C++ results in Table~\ref{tab:fixed}.','tab:depthhistory','@{}lrrrr@{}',
      [r'$m/s$','Scout DC',r'$U$ (\%)',r'Recall (\%)','S'],
      [[f"{r['routes']}/{r['quota']}",f"{r['mean_scout_dc']:.2f}",f"{r['quota_before_seed_percent']:.2f}",f"{r['recall_percent']:.4f}",r['severe']] for r in depth['rows']],wide=False)

(HERE/'TABLE-SOURCES.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'generated_tables':13,'frozen_source_files':len(sources)}))
