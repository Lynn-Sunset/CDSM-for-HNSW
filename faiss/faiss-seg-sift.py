# faiss 按 ES 生产形态跑 SIFT：32 连续分片（各 31250 节点）、逐分片 IndexHNSWFlat(M=16,efC=100)、
# 每查询逐分片搜索(ef_sh)后跨分片并集 top-10（同 PhaseSIFTSeg 的 siftseg-500 口径：前 500 官方查询）。
# 输出 recall@10 / 总 dc / 串行墙钟 p50，与 Java 分段数字（single 0.955@8234dc 6.6ms、mkL-R4 0.997@40564dc 29.7ms）对齐。
import faiss
import numpy as np
import struct
import time
from pathlib import Path

OUT = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out')
SIFT_DIR = Path(r'C:\Users\liuruilin\.cache\huggingface\hub\datasets--qbo-odp--sift1m\snapshots\bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b')
NSHARD = 32
NQ = 500
K = 10
SHARD_SIZE = 31250

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

print('[faiss-seg-sift] loading base/queries/gt ...')
base = load_fvecs(SIFT_DIR / 'sift_base.fvecs')
queries = load_fvecs(SIFT_DIR / 'sift_query.fvecs', NQ)
gt = load_ivecs(SIFT_DIR / 'sift_groundtruth.ivecs', NQ)
print(f'base={base.shape} queries={queries.shape}')

print('[faiss-seg-sift] building/loading 32 shard indexes ...')
t0 = time.time()
shards = []
for s in range(NSHARD):
    f = OUT / f'faiss-sift-seg-{s}.index'
    if f.exists():
        shards.append(faiss.read_index(str(f)))
    else:
        idx = faiss.IndexHNSWFlat(128, 16)
        idx.hnsw.efConstruction = 100
        idx.add(base[s*SHARD_SIZE:(s+1)*SHARD_SIZE])
        faiss.write_index(idx, str(f))
        shards.append(idx)
print(f'  {NSHARD} shards ready in {time.time()-t0:.0f}s')

# warmup
for s in shards:
    s.hnsw.efSearch = 400
for i in range(50):
    for s in shards:
        s.search(queries[i:i+1], K)

def query_once(i, ef_sh):
    best = {}
    for si, s in enumerate(shards):
        s.hnsw.efSearch = ef_sh
        D, I = s.search(queries[i:i+1], K)
        for lbl, d in zip(I[0], D[0]):
            if lbl < 0:
                continue
            g = si * SHARD_SIZE + lbl
            if g not in best or d < best[g]:
                best[g] = d
    top = sorted(best.items(), key=lambda kv: kv[1])[:K]
    return set(g for g, _ in top)

print('[faiss-seg-sift] ef_sh sweep (500 queries):')
rows = []
for ef_sh in [16, 32, 64, 100, 257, 400, 1600]:
    faiss.cvar.hnsw_stats.reset()
    times = []
    hits = 0
    for i in range(NQ):
        t0q = time.perf_counter_ns()
        top = query_once(i, ef_sh)
        t1q = time.perf_counter_ns()
        times.append(t1q - t0q)
        hits += len(top & set(gt[i]))
    ndis = faiss.cvar.hnsw_stats.ndis / NQ
    recall = hits / (NQ * K)
    times.sort()
    p50 = times[len(times)//2] / 1e6
    p90 = times[int(len(times)*0.9)] / 1e6
    rows.append((ef_sh, recall, ndis, p50, p90))
    print(f'  ef_sh={ef_sh:5d} recall@10={recall:.4f} total_dc={ndis:.0f} p50={p50:.2f}ms p90={p90:.2f}ms')

with open(OUT / 'faiss-seg-sift.csv', 'w') as f:
    f.write('efShard,recall,dcPerQuery,p50ms,p90ms\n')
    for ef_sh, r, d, p50, p90 in rows:
        f.write(f'{ef_sh},{r:.4f},{d:.0f},{p50:.3f},{p90:.3f}\n')
print('[faiss-seg-sift] wrote out/faiss-seg-sift.csv')
