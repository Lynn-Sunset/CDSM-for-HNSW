# 论文图表 v2.0 —— 全部基于真实引擎端到端与最新 sweep 数据（旧堆内 T2I 数据已弃用）
# fig1: GIST "门即差距"阶梯（dc vs recall：门控单束/CDSM门控/mkU深轮/singleU/faiss）
# fig2: T2I 真实引擎（门控平台 vs singleU 无墙 vs faiss 建图墙）
# fig3: Δ=α(1−P) 两库真实数据回归（GIST + T2I 真实引擎）
# fig4: 分段 SIFT 生产形态（Java CDSM 阶梯 vs faiss 逐分片）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\figs')
OUT.mkdir(exist_ok=True)
DATA = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

def means(csv, cols):
    df = pd.read_csv(DATA / csv)
    return {c: (df[c].mean() if c in df.columns else np.nan) for c in cols}

# ================= Fig 1: GIST the-gate-is-the-gap ladder =================
g = means('phaseGIST-latency.csv', [
    'single-1600-rec', 'single-1600-dc',
    'mkL-R4x400-rec', 'mkL-R4x400-dc', 'mkL-R8x400-rec', 'mkL-R8x400-dc',
    'mkL-R16x400-rec', 'mkL-R16x400-dc',
    'mkU-R2x3200-rec', 'mkU-R2x3200-dc', 'mkU-R2x6400-rec', 'mkU-R2x6400-dc',
    'mkU-R8x3200-rec', 'mkU-R8x3200-dc',
    'singleU-1600-rec', 'singleU-1600-dc', 'singleU-6400-rec', 'singleU-6400-dc',
    'singleU-12800-rec', 'singleU-12800-dc', 'singleU-25600-rec', 'singleU-25600-dc'])
faiss_g = pd.read_csv(DATA / 'faiss-gist.csv')   # efSearch,recall,dcPerQuery

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot([g['single-1600-dc']], [g['single-1600-rec']], 'kx', ms=10,
        label='gated single (0.397 @ 414 dc, any budget)')
ax.plot([g['mkL-R4x400-dc'], g['mkL-R8x400-dc'], g['mkL-R16x400-dc']],
        [g['mkL-R4x400-rec'], g['mkL-R8x400-rec'], g['mkL-R16x400-rec']],
        's--', color='tab:orange', label='CDSM gated (engine-compatible)')
ax.plot([g['mkU-R2x3200-dc'], g['mkU-R2x6400-dc'], g['mkU-R8x3200-dc']],
        [g['mkU-R2x3200-rec'], g['mkU-R2x6400-rec'], g['mkU-R8x3200-rec']],
        '^--', color='tab:green', label='CDSM deep rounds (mkU)')
ax.plot([g['singleU-1600-dc'], g['singleU-6400-dc'], g['singleU-12800-dc'],
         g['singleU-25600-dc']],
        [g['singleU-1600-rec'], g['singleU-6400-rec'], g['singleU-12800-rec'],
         g['singleU-25600-rec']],
        'o-', color='tab:red', label='singleU (gate deleted)')
ax.plot(faiss_g.dcPerQuery, faiss_g.recall, 'd-', color='tab:blue',
        label='faiss IndexHNSWFlat (ef sweep)')
ax.set_xscale('log')
ax.set_xlabel('distance computations per query (dc)')
ax.set_ylabel('Recall@10')
ax.set_title('GIST1M (960-d, M=16 efC=100): the gate is the gap')
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(OUT / 'fig1-gate-gap-gist.png', bbox_inches='tight'); plt.close(fig)

# ================= Fig 2: T2I real engine, no wall =================
# NOTE (2026-08-26 audit): phaseT2I-endtoend.csv stores real-engine recall (-rec)
# but NO per-query dc column. The dc values are the harness-reported visitedCounts
# (verified against the cfg run; single@400=0.477@360, single@1600=0.543@402 are the
# gated single; CDSM ladder from the full T2I rerun). faiss uses its own dcPerQuery.
# CDSM gated (mkL) and deep-round (mkU) are multi-point ladders from the 2026-08-26 rerun.
_t = means('phaseT2I-endtoend.csv', [
    'single-400-rec', 'single-1600-rec',
    'mkL-R1x400-rec', 'mkL-R2x400-rec', 'mkL-R4x400-rec', 'mkL-R8x400-rec', 'mkL-R16x400-rec',
    'mkU-R2x1600-rec', 'mkU-R4x1600-rec', 'mkU-R8x1600-rec', 'mkU-R4x6400-rec',
    'singleU-1600-rec', 'singleU-6400-rec', 'singleU-12800-rec', 'singleU-25600-rec'])
# (dc, recall) per config, from the T2I rerun console (dc = visitedCount; 16-distinct-entry fix)
mkl = [(730, _t['mkL-R1x400-rec']), (1086, _t['mkL-R2x400-rec']),
       (1804, _t['mkL-R4x400-rec']), (3207, _t['mkL-R8x400-rec']), (5921, _t['mkL-R16x400-rec'])]
mku = [(3560, _t['mkU-R2x1600-rec']), (6760, _t['mkU-R4x1600-rec']),
       (13157, _t['mkU-R8x1600-rec']), (25945, _t['mkU-R4x6400-rec'])]
singleu = [(1600, _t['singleU-1600-rec']), (6400, _t['singleU-6400-rec']),
           (12800, _t['singleU-12800-rec']), (25600, _t['singleU-25600-rec'])]
faiss_t = pd.read_csv(DATA / 'faiss-t2i.csv')   # efSearch,recall,dcPerQuery

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot([360, 402], [_t['single-400-rec'], _t['single-1600-rec']], 'kx', ms=10,
        label='gated single (0.543 @ 402 dc)')
ax.plot([p[0] for p in mkl], [p[1] for p in mkl], 's--', color='tab:orange',
        label='CDSM gated (mkL-R1..R16, engine-compatible)')
ax.plot([p[0] for p in mku], [p[1] for p in mku], '^--', color='tab:green',
        label='CDSM deep rounds (mkU-R2..R4x6400)')
ax.plot([p[0] for p in singleu], [p[1] for p in singleu], 'o-', color='tab:red',
        label='singleU (gate deleted) - no wall, 0.996')
ax.plot(faiss_t.dcPerQuery, faiss_t.recall, 'd--', color='tab:blue',
        label='faiss (build-side wall at 0.829)')
ax.set_xscale('log')
ax.set_xlabel('distance computations per query (dc)')
ax.set_ylabel('Recall@10')
ax.set_title('T2I 1M real engine (9.12 gen): gate-side plateau, no 0.83 wall')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(OUT / 'fig2-t2i-real-engine.png', bbox_inches='tight'); plt.close(fig)

# ================= Fig 3: Δ = α(1−P) on real data =================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
panels = [
    ('phaseGIST-latency.csv', 'single-1600-rec', 'mkL-R16x400-rec',
     'GIST1M (gain = CDSM-R16 − single@1600)'),
    ('phaseT2I-endtoend.csv', 'single-1600-rec', 'mkL-R4x400-rec',
     'T2I real engine (gain = CDSM-R4 − single@1600)'),
]
for ax, (csv, srec, mrec, title) in zip(axes, panels):
    df = pd.read_csv(DATA / csv)
    x = 1 - df[srec]
    y = df[mrec] - df[srec]
    ax.scatter(x, y, s=6, alpha=0.3)
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, coef[0] * xs + coef[1], 'r-',
            label=f'gain = {coef[0]:.3f}·(1−P) {coef[1]:+.3f}')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('1 − P (plateau gap)')
    ax.set_ylabel('CDSM gain')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / 'fig3-reg-real.png', bbox_inches='tight'); plt.close(fig)

# ================= Fig 4: segmented SIFT production shape =================
# (means from pvlb-execution-log.md: Java ladder vs faiss per-shard, 500 queries)
java_dc = [8234, 16709, 21703, 24955, 40564]
java_rec = [0.955, 0.985, 0.990, 0.994, 0.997]
java_lab = ['single@400', 'CDSM-R1×400', 'CDSM-R4×400-SG', 'CDSM-R2×400', 'CDSM-R4×400']
faiss_seg = pd.read_csv(DATA / 'faiss-seg-sift2.csv')   # efShard,recall,dcPerQuery,perQueryMs

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(java_dc, java_rec, 's--', color='tab:orange', label='Lucene gated + CDSM (per-shard)')
for x, y, l in zip(java_dc, java_rec, java_lab):
    ax.annotate(l, (x, y), textcoords='offset points', xytext=(4, -10), fontsize=7.5)
ax.plot(faiss_seg.dcPerQuery, faiss_seg.recall, 'd-', color='tab:blue',
        label='faiss per-shard (ef sweep)')
ax.set_xscale('log')
ax.set_xlabel('dc per query (sum over 32 shards)')
ax.set_ylabel('Recall@10 (global GT)')
ax.set_title('Segmented SIFT (32 shards): the honest production-shape comparison')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig(OUT / 'fig4-segmented-sift.png', bbox_inches='tight'); plt.close(fig)

print('v2.0 figures written to', OUT)
