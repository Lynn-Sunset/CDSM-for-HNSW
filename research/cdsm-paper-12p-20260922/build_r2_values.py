"""Numbers shared by the abstract, main text and latest-wave tables."""
from pathlib import Path
import json

R=Path(__file__).resolve().parent
S=json.loads((R/'source-data/cdsm-review-strengthening-20260922/SUMMARY.json').read_text(encoding='utf-8'))
M={(x['graph'],x['policy'],x['cap']):x for x in S['cells']}
G=['Faiss-M32','Imported-s42','Imported-s777','Imported-s1234']
def changes(control):
    return [100*(M[g,'F',10240]['qps']/M[g,S['selection'][g+'-10240']if control=='MEP_DEV'else control,10240]['qps']-1) for g in G]
def signed_range(values):return rf'${min(values):+.2f}\%$ to ${max(values):+.2f}\%$'
out=['% Generated only from the completed latest server control wave.']
for control,name in [('C','rTwoQpsC'),('MEP_DEV','rTwoQpsDev'),('FRONTIER','rTwoQpsFrontier'),('MEP16','rTwoQpsSixteen')]:
    out.append('\\newcommand{\\'+name+'}{'+signed_range(changes(control))+'}')
c=next(x for x in S['query_cluster_descriptive']if x['baseline']=='MEP_DEV')
v=c['recall_delta_pp'];lo,hi=c['recall_delta_ci95_pp']
out.append(r'\newcommand{\rTwoDevCluster}{'+rf'${v:+.3f}$ percentage points, with a descriptive query-cluster 95\% interval of $[{lo:+.3f},{hi:+.3f}]$'+'}')
(R/'paper/tables/r2-values.tex').write_text('\n'.join(out)+'\n',encoding='utf-8')
