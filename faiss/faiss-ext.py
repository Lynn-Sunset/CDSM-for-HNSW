# faiss 工业基线扩展 v2：SIFT1M / T2I 1M，同建图(M=16,efC=100)，同 2000 查询口径
# （与 Java PhaseSIFTLatency/PhaseT2ILatency 的 siftlat-2000/t2ilat-2000 对齐：
#   取查询文件前 2000 条 + 对应 GT 前 2000 行）。
# 输出：recall@10 vs dc(ndis) + 每查询墙钟 p50/p90（us/dc 单价），CSV 落盘。
import faiss
import numpy as np
import struct
import time
from pathlib import Path

OUT = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out')
NQ = 2000
K = 10

def load_fbin(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
        dim = struct.unpack('<i', f.read(4))[0]
        return np.frombuffer(f.read(4 * n * dim), dtype=np.float32).reshape(n, dim)

def load_ibin(p, k):
    with open(p, 'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
        kk = struct.unpack('<i', f.read(4))[0]
        return np.frombuffer(f.read(4 * n * kk), dtype=np.int32).reshape(n, kk)[:, :k]

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

def sweep(name, idx, queries, gt, efs):
    print(f'[{name}] efSearch sweep (2000 queries, warm first):')
    idx.hnsw.efSearch = efs[-1]
    for i in range(200):                       # warmup
        idx.search(queries[i:i+1], K)
    rows = []
    for ef in efs:
        idx.hnsw.efSearch = ef
        faiss.cvar.hnsw_stats.reset()
        times = []
        hits = 0
        for i in range(NQ):
            t0 = time.perf_counter_ns()
            D, I = idx.search(queries[i:i+1], K)
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)
            hits += len(set(I[0]) & set(gt[i]))
        ndis = faiss.cvar.hnsw_stats.ndis / NQ
        recall = hits / (NQ * K)
        times.sort()
        p50 = times[len(times)//2] / 1e6
        p90 = times[int(len(times)*0.9)] / 1e6
        us_per_dc = p50 * 1000.0 / ndis
        rows.append((ef, recall, ndis, p50, p90, us_per_dc))
        print(f'  ef={ef:5d} recall@10={recall:.4f} dc={ndis:.0f} '
              f'p50={p50:.2f}ms p90={p90:.2f}ms us/dc={us_per_dc:.3f}')
    with open(OUT / f'{name}.csv', 'w') as f:
        f.write('efSearch,recall,dcPerQuery,p50ms,p90ms,usPerDc\n')
        for ef, r, d, p50, p90, u in rows:
            f.write(f'{ef},{r:.4f},{d:.0f},{p50:.3f},{p90:.3f},{u:.4f}\n')
    print(f'[{name}] wrote out/{name}.csv')
    return rows

SIFT_DIR = Path(r'C:\Users\liuruilin\.cache\huggingface\hub\datasets--qbo-odp--sift1m\snapshots\bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b')
T2I_DIR = Path(r'C:\Users\liuruilin\.cache\huggingface\hub\datasets--unum-cloud--ann-t2i-1m\snapshots\94994f1ec2c2af1f41942f228b14d5d86f88fdec')

# ---- SIFT1M (L2, 128-d) ----
print('[faiss-sift] loading ...')
base = load_fvecs(SIFT_DIR / 'sift_base.fvecs')
queries = load_fvecs(SIFT_DIR / 'sift_query.fvecs', NQ)
gt = load_ivecs(SIFT_DIR / 'sift_groundtruth.ivecs', NQ)
print(f'base={base.shape} queries={queries.shape} gt={gt.shape}')
idx_file = OUT / 'faiss-sift-hnsw.index'
if idx_file.exists():
    idx = faiss.read_index(str(idx_file))
else:
    idx = faiss.IndexHNSWFlat(128, 16)
    idx.hnsw.efConstruction = 100
    idx.add(base)
    faiss.write_index(idx, str(idx_file))
sweep('faiss-sift', idx, queries, gt, [100, 400, 1600, 6400])

# ---- T2I (cosine via normalized L2, 200-d) ----
print('[faiss-t2i] loading ...')
base = load_fbin(T2I_DIR / 'base.1M.fbin')
queries = load_fbin(T2I_DIR / 'query.public.100K.fbin')[:NQ]
gt = load_ibin(T2I_DIR / 'groundtruth.public.100K.ibin', K)[:NQ]
base = base / np.linalg.norm(base, axis=1, keepdims=True)
queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
print(f'base={base.shape} queries={queries.shape} gt={gt.shape}')
idx_file = OUT / 'faiss-t2i-hnsw.index'
if idx_file.exists():
    idx = faiss.read_index(str(idx_file))
else:
    idx = faiss.IndexHNSWFlat(200, 16)
    idx.hnsw.efConstruction = 100
    idx.add(base)
    faiss.write_index(idx, str(idx_file))
sweep('faiss-t2i', idx, queries, gt, [100, 400, 1600, 6400])

print('[faiss-ext] done')
