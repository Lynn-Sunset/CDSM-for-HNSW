# 理论四图：平台成因 / CDSM 覆盖 / π 制度 / 尾部救援
# 图 T1-T2 为 2D 合成示意图（忠实于定理的机制）；T3-T4 为实测数据图。
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\figs')
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
BBOX = dict(fc='white', alpha=0.85, ec='none', pad=1.0)

rng = np.random.default_rng(7)

# ---------- 合成地形（图 T1/T2 共用） ----------
def cluster(center, sigma, n):
    return rng.normal(center, sigma, size=(n, 2))

c1 = cluster((0.0, 0.0), 1.15, 120)      # 主盆地（8 个 GT）
c2 = cluster((9.0, 1.2), 0.95, 60)       # B 盆地（1 个 GT）
c3 = cluster((4.6, 9.2), 0.95, 60)       # C 盆地（1 个 GT）
c4 = cluster((11.0, 8.8), 1.30, 80)      # D 荒地（0 个 GT）
q = np.array([0.3, 0.3])
e1 = c1[3]

def score(p, center=q, s2=3.5):
    d2 = ((p - center) ** 2).sum(axis=1)
    return 1.0 / (1.0 + d2 / s2)

# 水位线 θ = score 在半径 r_theta 处
r_theta = 3.4
theta = 1.0 / (1.0 + r_theta ** 2 / 3.5)

gt = {'c1': c1[:8], 'c2': c2[[7]], 'c3': c3[[13]]}

def draw_base(ax, title):
    ax.scatter(c4[:, 0], c4[:, 1], s=10, c='#c9c9c9', label='barren cluster D')
    ax.scatter(c1[:, 0], c1[:, 1], s=12, c='#7fb3d5', label='cluster A')
    ax.scatter(c2[:, 0], c2[:, 1], s=12, c='#f0b27a', label='cluster B')
    ax.scatter(c3[:, 0], c3[:, 1], s=12, c='#82e0aa', label='cluster C')
    # 簇内 ε-边（示意连通性）
    for pts, col in [(c1, '#7fb3d5'), (c2, '#f0b27a'), (c3, '#82e0aa')]:
        for i in range(0, len(pts), 3):
            d = np.linalg.norm(pts - pts[i], axis=1)
            nb = np.where((d > 0.1) & (d < 1.9))[0]
            for j in nb[:2]:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                        color=col, lw=0.6, alpha=0.5, zorder=1)
    ax.plot(*q, 'k*', ms=18, zorder=6)
    ax.annotate('query q', (q[0], q[1]), xytext=(1.8, -1.0),
                fontsize=12, fontweight='bold', bbox=BBOX)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('embedding dim 1'); ax.set_ylabel('embedding dim 2')
    # aspect auto: wider box so labels fit inside the frame

# ---------- Fig T1: 平台成因（定理 1） ----------
fig, ax = plt.subplots(figsize=(7.2, 6.2))
draw_base(ax, 'Theorem 1: the single beam is trapped in C(e, θ)')
t = np.linspace(0, 2 * np.pi, 200)
ax.plot(q[0] + r_theta * np.cos(t), q[1] + r_theta * np.sin(t),
        'k--', lw=1.5, label='water level θ (gate: score ≥ θ only)')
ax.annotate('C(e,θ): covered\ncomponent', (0.4, 1.3), fontsize=11, color='#1f618d',
            ha='center', fontweight='bold', bbox=BBOX)
ax.plot(*e1, 'v', ms=13, c='#1f618d', zorder=6)
ax.annotate('entry e', (e1[0] - 3.0, e1[1] - 1.6), fontsize=11, color='#1f618d', bbox=BBOX)
for key, col in [('c1', '#f5b7b1'), ('c2', '#f9e79f'), ('c3', '#a9dfbf')]:
    for p in gt[key]:
        ax.plot(*p, '*', ms=14, c=col, mec='k', mew=0.6, zorder=7)
ax.annotate('8 GT found', (-4.4, 4.9), fontsize=11, color='#c0392b', fontweight='bold', ha='left', bbox=BBOX)
ax.annotate('1 GT — below θ:\nunreachable no\nmatter the budget', (9.0, 3.0),
            fontsize=10, color='#7d6608', ha='center', bbox=BBOX)
ax.annotate('1 GT — another basin', (7.6, 11.8), fontsize=10, color='#1e8449', ha='center', bbox=BBOX)
ax.annotate('plateau ceiling =\n|GT ∩ C(e,θ)| / k', xy=(0.4, -3.0), fontsize=11,
            color='#1f618d', fontweight='bold', bbox=BBOX)
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(-4.5, 14); ax.set_ylim(-4.5, 13.5)
fig.tight_layout(); fig.savefig(OUT / 'figT1-plateau-why.png'); plt.close(fig)

# ---------- Fig T2: CDSM 覆盖不交并（定理 2） ----------
fig, ax = plt.subplots(figsize=(7.2, 6.2))
draw_base(ax, 'Theorem 2: CDSM = disjoint union of covered components')
t = np.linspace(0, 2 * np.pi, 200)
ax.plot(q[0] + r_theta * np.cos(t), q[1] + r_theta * np.sin(t),
        'k--', lw=1.5, label='probe water level θ')
ax.plot(*e1, 'v', ms=13, c='#1f618d', zorder=6)
ax.annotate('probe: covers A, marks it', (e1[0] + 0.2, e1[1] + 0.4), fontsize=10, color='#1f618d', bbox=BBOX)
e2 = c2[9]; e3 = c3[11]
for e, col, lab, off, ha in [(e2, '#e67e22', 'round 1: entry in B\n(never marked)', (4.8, -2.0), 'right'),
                             (e3, '#27ae60', 'round 2: entry in C\n(never marked)', (1.2, 2.2), 'left')]:
    ax.plot(*e, '^', ms=14, c=col, zorder=6)
    ax.annotate(lab, (e[0], e[1]), xytext=(e[0] + off[0], e[1] + off[1]),
                fontsize=10, color=col, ha=ha, bbox=BBOX)
    circle = plt.Circle(e, 1.4, color=col, alpha=0.18, zorder=2)
    ax.add_patch(circle)
ax.annotate('round 3: only barren D\nunmarked — SG stops\n(score-gated early stop)',
            (13.8, 5.8), fontsize=10, color='#7f8c8d', ha='right', bbox=BBOX)
circle = plt.Circle((11.0, 8.8), 1.8, color='#95a5a6', alpha=0.12, linestyle=':',
                    edgecolor='#7f8c8d', zorder=2)
ax.add_patch(circle)
for key, col in [('c1', '#f5b7b1'), ('c2', '#f9e79f'), ('c3', '#a9dfbf')]:
    for p in gt[key]:
        ax.plot(*p, '*', ms=14, c=col, mec='k', mew=0.6, zorder=7)
ax.annotate('all 10 GT covered:\nunion of disjoint components', (4.6, 3.4),
            fontsize=12, color='#c0392b', fontweight='bold', ha='center', bbox=BBOX)
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(-4.5, 14); ax.set_ylim(-4.5, 13.5)
fig.tight_layout(); fig.savefig(OUT / 'figT2-cdsm-coverage.png'); plt.close(fig)

# ---------- Fig T3: π 制度（定理 4）——真实数据版：逐查询每轮增益 ----------
# NOTE (2026-08-26 audit): the T2I panel is DROPPED. The real-engine T2I CSV has
# only the R4 config (no R1/R2), and the only R1/R2/R4 T2I series (t2ilat-r2) is
# the superseded in-heap biased-build data. No valid multimodal R1/R2/R4 gain
# source exists, so the figure shows the two VALID regimes (concentrated vs spread).
import pandas as pd
import matplotlib.pyplot as plt
DATA = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out')

def round_gains(csv_file, base_col, r1_col, r2_col, r4_col):
    df = pd.read_csv(DATA / csv_file)
    g1 = df[r1_col] - df[base_col]
    g2 = df[r2_col] - df[r1_col]
    g34 = (df[r4_col] - df[r2_col]) / 2.0
    return [g1, g2, g34]

med = round_gains('phaseD-sweep-sweep2-dense.csv', 'single-400-rec',
                  'mk-R1x400-rec', 'mk-R2x400-rec', 'mk-R4x400-rec')
sift = round_gains('phaseSIFT-latency-siftlat-2000.csv', 'single-1600-rec',
                   'mkL-R1x400-rec', 'mkL-R2x400-rec', 'mkL-R4x400-rec')

fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), sharey=True)
labels = ['round 1', 'round 2', 'rounds 3-4 (avg)']
for ax, data, name, col in [
        (axes[0], med, 'medical (concentrated \u03c0)', '#1f618d'),
        (axes[1], sift, 'SIFT1M (spread \u03c0)', '#c0392b')]:
    means = [g.mean() for g in data]
    ax.bar(labels, means, color=col, alpha=0.75)
    for i, g in enumerate(data):
        jitter = np.random.default_rng(3 + i).uniform(-0.18, 0.18, len(g))
        ax.plot(i + jitter, g, '.', ms=3, color='#555555', alpha=0.25)
        ax.text(i, means[i] + 0.012, f'{means[i]:+.3f}', ha='center', fontsize=11,
                fontweight='bold')
    ax.set_title(name, fontsize=11)
    ax.grid(alpha=0.3, axis='y')
axes[0].set_ylabel('per-round recall gain \u0394')
# zoom the shared y-axis onto the mean ladder (per-query jitter is noisy and
# extends far beyond; it is clipped so the mean gains are readable)
axes[0].set_ylim(-0.03, 0.13)
fig.suptitle('Proposition 1 (real data): per-round gains --- concentrated \u03c0 decays '
             '(medical), spread \u03c0 is flat (SIFT); dots = per-query gains',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT / 'figT3-pi-regimes.png', bbox_inches='tight'); plt.close(fig)

# ---------- Fig T4: 尾部救援（定理 5，实测点） ----------
# NOTE (2026-08-26 audit): the old single-panel version plotted a T2I not-rescued
# series [76,70,62,54] = 100 - [24,30,38,46], which is the SUPERSEDED in-heap
# T2I safety-tail series. It is replaced by a two-panel honest plot:
#   (A) SIFT per-round full-rescue (the real 48->61->75->88 ladder; Prop-2
#       multiplicative falloff, exponential fit) -- the one corpus with valid
#       per-round data.
#   (B) config-level full-rescue at [single@1600, CDSM-Lite-R4x400 (gated),
#       mkU-R4x1600 (deep round)] -- distinguishes the gated vs deep-round tiers
#       the paper's Sec 6.5 highlights, with the REAL T2I values (34/63/98).
rounds = np.array([0, 1, 2, 4])
sift_nr = np.array([52, 39, 25, 12])   # not-fully-rescued % = 100 - [48,61,75,88]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
ax = axes[0]
ax.plot(rounds, sift_nr, 'o-', ms=8, color='#c0392b', label='SIFT1M (not rescued, %)')
b, a = np.polyfit(rounds, np.log(sift_nr), 1)
xf = np.linspace(0, 4, 50)
ax.plot(xf, np.exp(a + b * xf), '--', color='#c0392b', alpha=0.6,
        label=f'SIFT fit: e^({a:.2f}{b:+.2f} R)')
ax.set_xticks(rounds); ax.set_xticklabels(['Single@1600', 'CDSM-R1', 'CDSM-R2', 'CDSM-R4'])
ax.set_yscale('log')
ax.set_ylabel('% catastrophic NOT fully rescued (log)')
ax.set_xlabel('rounds R (gated CDSM ladder)')
ax.set_title('(A) Prop. 2: falls multiplicatively in R', fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

ax = axes[1]
configs = ['single@1600', 'CDSM-Lite-R4x400\n(gated)', 'mkU-R4x1600\n(deep round)']
x = np.arange(3)
t2i = [34, 63, 98]
sift = [48, 88, np.nan]
ax.bar(x - 0.2, sift, width=0.4, color='#e67e22', alpha=0.8, label='SIFT1M')
ax.bar(x + 0.2, t2i, width=0.4, color='#1f618d', alpha=0.8, label='T2I')
for xi, v in [(x[0]-0.2, sift[0]), (x[1]-0.2, sift[1]), (x[0]+0.2, t2i[0]),
              (x[1]+0.2, t2i[1]), (x[2]+0.2, t2i[2])]:
    if not np.isnan(v):
        ax.text(xi, v + 2, f'{v:.0f}%', ha='center', fontsize=8.5, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8.5)
ax.set_ylabel('full-rescue rate (% of catastrophes)')
ax.set_ylim(0, 112)
ax.set_title('(B) gated vs deep round (real data)', fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
fig.tight_layout(); fig.savefig(OUT / 'figT4-tail-rescue.png', bbox_inches='tight'); plt.close(fig)

print('theory figs written to', OUT)
