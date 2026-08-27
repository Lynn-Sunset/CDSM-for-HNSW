# 真实数据版 T1/T2（最终版）：SIFT q2130 灾难查询的"解剖学"图。
# 布局 = 本地节点（single+r+gt+entries+desc，无 ctx/gtnb）向量的 PCA 投影；
# 语义 = 颜色（visited / rounds / GT 判定）+ 图题中的严格验证数字：
#   G（全图）0/10 断连；G_θ（score>θ 边）4/10 断连 —— 恰好是 4 个漏检 GT。
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\figs')
DATA = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

q_idx = '2130'
nodes = pd.read_csv(DATA / f'realvis-q{q_idx}-nodes.csv')
edges = pd.read_csv(DATA / f'realvis-q{q_idx}-edges.csv')
meta = {}
for line in (DATA / f'realvis-q{q_idx}-meta.txt').read_text().splitlines():
    k, _, v = line.partition('=')
    meta[k] = v
theta = float(meta['probeTheta'])

# ---- connectivity verdicts (strict semantics, real) ----
tagA = nodes['tag'].astype(str).values
ordA = nodes['ord'].astype(int).values
scA = dict(zip(ordA, nodes['score'].astype(float).values))
G = nx.Graph(); G.add_nodes_from(ordA)
G.add_edges_from(edges[['u', 'v']].values)
Gt = nx.Graph(); Gt.add_nodes_from(ordA)
for u, v in edges[['u', 'v']].values:
    if scA[u] > theta and scA[v] > theta:
        Gt.add_edge(int(u), int(v))
single_set = set(ordA[tagA == 'single'])
gt_ords = sorted(set(ordA[tagA == 'gt']))
comp_single = next(c for c in nx.connected_components(Gt)
                   if any(o in single_set for o in set(c) & single_set))
vGt = ['visited' if o in single_set else
       ('isolated' if o not in comp_single else 'near-main') for o in gt_ords]
n_iso = sum(1 for v in vGt if v == 'isolated')
n_vis = sum(1 for v in vGt if v == 'visited')

# ---- vectors (vec CSV: q/gt/ctx/single/r1-r4/entries/desc) ----
vdf = pd.read_csv(DATA / f'realvis-q{q_idx}.csv')
vtags = vdf['tag'].astype(str).values
vords = vdf['ord'].astype(int).values
V = vdf[[c for c in vdf.columns if c.startswith('v')]].values.astype(float)
keep = vtags != 'ctx'
X = V[keep]; kt = vtags[keep]; ko = vords[keep]
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
Z = (U[:, :2] * S[:2] * np.sqrt(len(X)))
zmap = {o: Z[i] for i, o in enumerate(ko)}
print(f'PCA basis: {len(X)} local points; 2D variance {sum(S[:2]**2)/sum(S**2)*100:.1f}%')

def draw(figpath, title, mode):
    fig, ax = plt.subplots(figsize=(8.2, 6.6), constrained_layout=True)
    if mode == 't1':
        mm = np.array([(t not in ('single', 'gt')) for t in kt])
        ax.scatter(Z[mm][:, 0], Z[mm][:, 1], s=6, c='#cfcfcf', alpha=0.75,
                   label='rounds + entries + descended')
    else:
        for ph, col, lab in [('r1', '#e67e22', 'round 1 (new)'), ('r2', '#27ae60', 'round 2 (new)'),
                             ('r3', '#8e44ad', 'rounds 3-4 (new)')]:
            mm = np.array([t == ph for t in kt])
            if mm.any():
                ax.scatter(Z[mm][:, 0], Z[mm][:, 1], s=7, c=col, alpha=0.7, label=lab)
    ms = np.array([t == 'single' for t in kt])
    ax.scatter(Z[ms][:, 0], Z[ms][:, 1], s=8, c='#3498db', alpha=0.8,
               label='probe / single@400 visited (293 nodes)')
    zlo_y, zhi_y = Z[:, 1].min(), Z[:, 1].max()
    for li, (o, v) in enumerate(zip(gt_ords, vGt)):
        z = zmap.get(o)
        if z is None:
            continue
        col = '#27ae60' if v == 'visited' else ('#c0392b' if v == 'isolated' else '#f39c12')
        ax.plot(z[0], z[1], '*', ms=20, c=col, mec='k', mew=0.8, zorder=10)
        fy = (z[1] - zlo_y) / (zhi_y - zlo_y)
        if fy < 0.22:
            dy = 12
        elif fy > 0.85:
            dy = -16
        else:
            dy = -16 if li % 2 == 0 else 12
        dx, ha = (7, 'left') if li % 4 < 2 else (-7, 'right')
        ax.annotate(f'GT {o}', (z[0], z[1]), xytext=(dx, dy), textcoords='offset points',
                    fontsize=7.5, color=col, ha=ha,
                    bbox=dict(fc='white', alpha=0.8, ec='none', pad=0.6))
    for k in range(1, 5):
        eo = vords[vtags == f'entry{k}']
        if len(eo):
            e = int(eo[0])
            if e in zmap:
                ax.plot(*zmap[e], '^', ms=13, c='#b7950b', mec='k', mew=0.5, zorder=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel('PCA dim 1 (real 128-d vectors, local points)')
    ax.set_ylabel('PCA dim 2')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.savefig(figpath, bbox_inches='tight')
    plt.close(fig)

draw(OUT / 'figT1-plateau-why.png',
     f'Theorem 1 (REAL, SIFT query #{q_idx}): gate theta={theta:.0f} cuts the valleys\n'
     f'in G: 0/10 GT disconnected — in G_theta: {n_iso}/10 GT isolated islands (exactly the missed)',
     't1')
draw(OUT / 'figT2-cdsm-coverage.png',
     f'Theorem 2 (REAL, SIFT query #{q_idx}): rounds re-open the gate on new territory\n'
     f'probe {meta.get("singleVisited")} nodes + rounds {meta.get("round1New")}/{meta.get("round2New")}/'
     f'{meta.get("round3New")}/{meta.get("round4New")} — GT rescued by rounds: {meta.get("gtInRoundsOnly")}',
     't2')
print('PCA-anatomy figures written (no synthetic data anywhere)')
