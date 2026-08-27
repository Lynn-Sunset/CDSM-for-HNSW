# GIST1M: faiss IndexHNSWFlat(M=16, efC=100) 墙钟口径基线
# 与 faiss-gist.py 同索引缓存；单查询串行计时（faiss 单查询搜索本身单线程），
# 输出 per-query p50/p90 时延、dc(ndis)/query、µs/dc 单价。
import faiss
import numpy as np
import struct
import time
import os
from pathlib import Path

DATA = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\gist')

def load_fvecs(path, maxn=None):
    with open(path, 'rb') as f:
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

def load_ivecs_topk(path, k):
    with open(path, 'rb') as f:
        rows = []
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            kk = struct.unpack('<i', hdr)[0]
            arr = np.frombuffer(f.read(4 * kk), dtype=np.int32)
            rows.append(arr[:k])
    return np.vstack(rows)

print('[faiss-gist-lat] loading ...')
base = load_fvecs(DATA / 'gist_base.fvecs')
queries = load_fvecs(DATA / 'gist_query.fvecs')
gt = load_ivecs_topk(DATA / 'gist_groundtruth.ivecs', 10)
print(f'base={base.shape} queries={queries.shape}')

IDX_FILE = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\faiss-gist-hnsw.index')
idx = faiss.read_index(str(IDX_FILE))
print(f'[faiss-gist-lat] index loaded, size on disk = {IDX_FILE.stat().st_size/2**30:.2f} GiB, ntotal={idx.ntotal}')

# 内存口径: 向量数组 + 邻接表 (faiss HNSWFlat 全精度 fp32, 无量化)
nbytes_vec = idx.ntotal * 960 * 4
print(f'[faiss-gist-lat] resident estimate: vectors {nbytes_vec/2**30:.2f} GiB + graph ~{(IDX_FILE.stat().st_size - nbytes_vec)/2**30:.2f} GiB')

k = 10
efs = [100, 200, 400, 800, 1600, 3200]

# warmup: 200 queries at max ef (JIT/缓存预热, 与 Java 侧同思路)
idx.hnsw.efSearch = efs[-1]
for i in range(200):
    idx.search(queries[i:i+1], k)

rows = []
for ef in efs:
    idx.hnsw.efSearch = ef
    faiss.cvar.hnsw_stats.reset()
    times = []
    hits = 0
    for i in range(len(queries)):
        t0 = time.perf_counter_ns()
        D, I = idx.search(queries[i:i+1], k)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
        hits += len(set(I[0]) & set(gt[i]))
    ndis = faiss.cvar.hnsw_stats.ndis / len(queries)
    recall = hits / (len(queries) * k)
    times.sort()
    p50 = times[len(times)//2] / 1e6
    p90 = times[int(len(times)*0.9)] / 1e6
    mean = sum(times) / len(times) / 1e6
    us_per_dc = p50 * 1000.0 / ndis
    rows.append((ef, recall, ndis, p50, p90, mean, us_per_dc))
    print(f'  ef={ef:5d} recall@10={recall:.4f} dc={ndis:.0f} '
          f'p50={p50:.2f}ms p90={p90:.2f}ms mean={mean:.2f}ms '
          f'us/dc={us_per_dc:.3f}')

out = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\faiss-gist-latency.csv')
with open(out, 'w') as f:
    f.write('efSearch,recall,dcPerQuery,p50ms,p90ms,meanMs,usPerDc\n')
    for ef, r, d, p50, p90, m, u in rows:
        f.write(f'{ef},{r:.4f},{d:.0f},{p50:.3f},{p90:.3f},{m:.3f},{u:.4f}\n')
print(f'[faiss-gist-lat] wrote {out}')
