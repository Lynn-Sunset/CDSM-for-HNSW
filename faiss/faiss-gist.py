# GIST1M: faiss IndexHNSW(M=16, efC=100) 工业基线 — recall vs dc (ndis) 曲线。
# 与 PhaseGISTLatency 同建图参数、同 1000 官方查询（全量顺序），L2。
import faiss
import numpy as np
import struct
from pathlib import Path
import time

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

print('[faiss-gist] loading ...')
base = load_fvecs(DATA / 'gist_base.fvecs')          # 1M x 960
queries = load_fvecs(DATA / 'gist_query.fvecs')      # 1000 x 960
gt = load_ivecs_topk(DATA / 'gist_groundtruth.ivecs', 10)   # 1000 x 10
print(f'base={base.shape} queries={queries.shape} gt={gt.shape}')

t0 = time.time()
IDX_FILE = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\faiss-gist-hnsw.index')
if IDX_FILE.exists():
    idx = faiss.read_index(str(IDX_FILE))
    print(f'[faiss-gist] index loaded from cache')
else:
    idx = faiss.IndexHNSWFlat(960, 16)          # M=16, 与我们的建图一致
    idx.hnsw.efConstruction = 100               # efC=100
    idx.add(base)
    faiss.write_index(idx, str(IDX_FILE))
    print(f'[faiss-gist] built in {time.time()-t0:.0f}s (M=16 efC=100), index saved')

# 查询顺序 = 全部 1000 条（与 Java 全量顺序模式严格一致）
k = 10
print('[faiss-gist] efSearch sweep (recall@10 vs ndis):')
rows = []
for ef in [100, 200, 400, 800, 1600, 3200, 6400, 12800]:
    idx.hnsw.efSearch = ef
    faiss.cvar.hnsw_stats.reset()
    hits = 0
    for i in range(len(queries)):
        D, I = idx.search(queries[i:i+1], k)
        hits += len(set(I[0]) & set(gt[i]))
    ndis_total = faiss.cvar.hnsw_stats.ndis
    recall = hits / (len(queries) * k)
    rows.append((ef, recall, ndis_total / len(queries)))
    print(f'  ef={ef:6d}  recall@10={recall:.3f}  dc(ndis)/query={ndis_total/len(queries):.0f}')

with open(Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\faiss-gist.csv'), 'w') as f:
    f.write('efSearch,recall,dcPerQuery\n')
    for ef, r, d in rows:
        f.write(f'{ef},{r:.4f},{d:.0f}\n')
print('[faiss-gist] wrote out/faiss-gist.csv')
