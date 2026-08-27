# faiss 32 分片 SIFT 墙钟修正版：批量搜索（每分片 500 查询一次调用）、OMP_NUM_THREADS=1，
# 消除 Python 逐查询循环开销，得到与 Java 串行口径可比的单查询等价时延。
# dc 口径不变（ndis 累计 / 500）；recall 按逐查询并集计算。
import os
os.environ['OMP_NUM_THREADS'] = '1'
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

print('[faiss-seg-sift2] loading ...')
queries = load_fvecs(SIFT_DIR / 'sift_query.fvecs', NQ)
gt = load_ivecs(SIFT_DIR / 'sift_groundtruth.ivecs', NQ)

shards = [faiss.read_index(str(OUT / f'faiss-sift-seg-{s}.index')) for s in range(NSHARD)]
for s in shards:
    s.hnsw.efSearch = 400

# warmup: batch search all shards at ef=400
for s in shards:
    s.search(queries, K)

rows = []
for ef_sh in [16, 32, 64, 100, 257, 400]:
    faiss.cvar.hnsw_stats.reset()
    t0 = time.perf_counter_ns()
    per_shard = []
    for s in shards:
        s.hnsw.efSearch = ef_sh
        D, I = s.search(queries, K)          # 批量：NQ 查询一次调用
        per_shard.append((I, D))
    t1 = time.perf_counter_ns()
    total_ms = (t1 - t0) / 1e6
    ndis = faiss.cvar.hnsw_stats.ndis / NQ

    # 逐查询并集 recall
    hits = 0
    for i in range(NQ):
        best = {}
        for si, (I, D) in enumerate(per_shard):
            for lbl, d in zip(I[i], D[i]):
                if lbl < 0:
                    continue
                g = si * SHARD_SIZE + lbl
                if g not in best or d < best[g]:
                    best[g] = d
        top = set(g for g, _ in sorted(best.items(), key=lambda kv: kv[1])[:K])
        hits += len(top & set(gt[i]))
    recall = hits / (NQ * K)
    rows.append((ef_sh, recall, ndis, total_ms / NQ))
    print(f'  ef_sh={ef_sh:5d} recall@10={recall:.4f} total_dc={ndis:.0f} '
          f'per-query(seq-equiv)={total_ms/NQ:.2f}ms')

with open(OUT / 'faiss-seg-sift2.csv', 'w') as f:
    f.write('efShard,recall,dcPerQuery,perQueryMs\n')
    for ef_sh, r, d, ms in rows:
        f.write(f'{ef_sh},{r:.4f},{d:.0f},{ms:.3f}\n')
print('[faiss-seg-sift2] wrote out/faiss-seg-sift2.csv')
