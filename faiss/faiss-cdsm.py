# CDSM 协议移植到 faiss（CDSM-Lite 类比，独立束、无共享 visited）：
#   建图期从 level>=2 节点采样入口表 E，每查询 R 轮、每轮 ef_r 独立束、结果并集。
# 对照 = 同 ef_total 单束。输出 recall@10 / dc(ndis) / 墙钟 p50 / us/dc。
# 注意：faiss 无 visited 共享接口 -> 各轮独立探索（等价 Java mkU 且更弱），
#       dc 含跨轮重复打分，如实计入（这才是 faiss 的真实成本）。
import faiss
import numpy as np
import struct
import time
from pathlib import Path

OUT = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out')
K = 10

def load_fvecs(p, maxn=None):
    with open(p, 'rb') as f:
        xs = []
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            dim = struct.unpack('<i', hdr)[0]
            xs.append(np.frombuffer(f.read(4 * dim), dtype=np.float32))
            if maxn and len(xs) >= maxn:
                break
    return np.vstack(xs)

def load_ivecs(p, maxn=None):
    with open(p, 'rb') as f:
        rows = []
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            kk = struct.unpack('<i', hdr)[0]
            rows.append(np.frombuffer(f.read(4 * kk), dtype=np.int32)[:K])
            if maxn and len(rows) >= maxn:
                break
    return np.vstack(rows)

def load_fbin(p, maxn=None):
    with open(p, 'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
        dim = struct.unpack('<i', f.read(4))[0]
        x = np.frombuffer(f.read(4 * n * dim), dtype=np.float32).reshape(n, dim)
        return x[:maxn] if maxn else x

def load_ibin(p, k, maxn=None):
    with open(p, 'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
        kk = struct.unpack('<i', f.read(4))[0]
        x = np.frombuffer(f.read(4 * n * kk), dtype=np.int32).reshape(n, kk)[:, :k]
        return x[:maxn] if maxn else x

def build_entry_table(idx, e, seed=42):
    """建图期入口表: level>=2 节点中均匀采样 e 个（CDSM-Lite 类比）。"""
    lv = faiss.vector_to_array(idx.hnsw.levels)
    rng = np.random.default_rng(seed)
    cand = np.nonzero(lv >= 2)[0]
    if len(cand) < e:
        cand = np.arange(idx.ntotal)
    return rng.choice(cand, size=e, replace=False).tolist()

def run_one(idx, q, ef, entry, reset_stats=True):
    if reset_stats:
        faiss.cvar.hnsw_stats.reset()
    idx.hnsw.efSearch = ef
    idx.hnsw.entry_point = int(entry)
    D, I = idx.search(q, K)
    return I[0], D[0], faiss.cvar.hnsw_stats.ndis

def configs():
    """(label, rounds, ef_r) — 每轮 ef_r, 总 ef = R*ef_r"""
    out = []
    for total, rs in [(400, [1, 2, 4, 8, 16]), (1600, [1, 2, 4, 8, 16]), (6400, [1, 2, 4, 8, 16])]:
        for r in rs:
            ef_r = max(16, total // r)
            out.append((f'R{r}x{ef_r}', r, ef_r))
    return out

def sweep(name, idx, queries, gt, nq, entry_table):
    print(f'[faiss-cdsm:{name}] warmup 200 queries at R16x100 ...')
    idx.hnsw.efSearch = 100
    for i in range(200):
        idx.search(queries[i:i+1], K)
    results = {}
    for label, R, ef_r in configs():
        faiss.cvar.hnsw_stats.reset()
        times = []
        hits = 0
        for i in range(nq):
            t0 = time.perf_counter_ns()
            best = {}
            for r in range(R):
                I, D, ndis = run_one(idx, queries[i:i+1], ef_r, entry_table[(i + r) % len(entry_table)], reset_stats=False)
                for lbl, d in zip(I, D):
                    if lbl < 0:
                        continue
                    if lbl not in best or d < best[lbl]:
                        best[lbl] = d
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)
            top = sorted(best.items(), key=lambda kv: kv[1])[:K]
            hits += len(set(l for l, _ in top) & set(gt[i]))
        ndis = faiss.cvar.hnsw_stats.ndis / nq
        recall = hits / (nq * K)
        times.sort()
        p50 = times[len(times)//2] / 1e6
        p90 = times[int(len(times)*0.9)] / 1e6
        us_per_dc = p50 * 1000.0 / ndis
        results[label] = (recall, ndis, p50, p90, us_per_dc)
        print(f'  {label:8s} recall@10={recall:.4f} dc={ndis:.0f} '
              f'p50={p50:.2f}ms p90={p90:.2f}ms us/dc={us_per_dc:.3f}')
    with open(OUT / f'faiss-cdsm-{name}.csv', 'w') as f:
        f.write('config,recall,dcPerQuery,p50ms,p90ms,usPerDc\n')
        for label, (r, d, p50, p90, u) in results.items():
            f.write(f'{label},{r:.4f},{d:.0f},{p50:.3f},{p90:.3f},{u:.4f}\n')
    print(f'[faiss-cdsm:{name}] wrote out/faiss-cdsm-{name}.csv')

GIST = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\gist')
SIFT_DIR = Path(r'C:\Users\liuruilin\.cache\huggingface\hub\datasets--qbo-odp--sift1m\snapshots\bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b')

if 'gist' in __import__('sys').argv[1:]:
    print('[faiss-cdsm:gist] loading ...')
    queries = load_fvecs(GIST / 'gist_query.fvecs')
    gt = load_ivecs(GIST / 'gist_groundtruth.ivecs')
    idx = faiss.read_index(str(OUT / 'faiss-gist-hnsw.index'))
    table = build_entry_table(idx, 32)
    sweep('gist', idx, queries, gt, len(queries), table)

if 'sift' in __import__('sys').argv[1:]:
    print('[faiss-cdsm:sift] loading ...')
    nq = 2000
    queries = load_fvecs(SIFT_DIR / 'sift_query.fvecs', nq)
    gt = load_ivecs(SIFT_DIR / 'sift_groundtruth.ivecs', nq)
    idx = faiss.read_index(str(OUT / 'faiss-sift-hnsw.index'))
    table = build_entry_table(idx, 32)
    sweep('sift', idx, queries, gt, nq, table)

if 't2i' in __import__('sys').argv[1:]:
    T2I_DIR = Path(r'C:\Users\liuruilin\.cache\huggingface\hub\datasets--unum-cloud--ann-t2i-1m\snapshots\94994f1ec2c2af1f41942f228b14d5d86f88fdec')
    print('[faiss-cdsm:t2i] loading ...')
    nq = 2000
    queries = load_fbin(T2I_DIR / 'query.public.100K.fbin', nq)
    gt = load_ibin(T2I_DIR / 'groundtruth.public.100K.ibin', K, nq)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    idx = faiss.read_index(str(OUT / 'faiss-t2i-hnsw.index'))
    table = build_entry_table(idx, 32)
    sweep('t2i', idx, queries, gt, nq, table)

print('[faiss-cdsm] done')
