# LAION-CLIP slice: faiss IndexHNSWFlat(M=16, efC=100) baseline.
# base and queries: float32 fvecs (dim read from data); cosine = unit-norm + L2.
# recall@10 vs engine-space GT (out/laion-gt-engine.ivecs, Java BIG_ENDIAN format);
# if absent, brute-force GT is computed here and cached in that same format.
import faiss
import numpy as np
import struct
import time
from pathlib import Path

OUT = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out')
LAION = OUT / 'laion'
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

print('[faiss-laion] loading ...')
base = load_fvecs(LAION / 'laion_base.fvecs')
queries = load_fvecs(LAION / 'laion_query.fvecs')
dim = base.shape[1]
print(f'base={base.shape} queries={queries.shape} dim={dim}')

base = base / np.linalg.norm(base, axis=1, keepdims=True)
queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

GT = OUT / 'laion-gt-engine.ivecs'
def load_gt():
    with open(GT, 'rb') as f:
        import io
        data = f.read()
    nq = struct.unpack('>i', data[0:4])[0]
    if nq != len(queries):
        return None
    off = 4; gt = []
    for q in range(nq):
        ln = struct.unpack('>i', data[off:off+4])[0]; off += 4
        gt.append(list(struct.unpack('>%di' % ln, data[off:off+4*ln]))); off += 4*ln
    return gt

if GT.exists():
    gt = load_gt()
    if gt is not None:
        print('[faiss-laion] GT loaded from cache')
    else:
        print('[faiss-laion] GT cache nq mismatch -> recompute')
else:
    gt = None
if gt is None:
    print('[faiss-laion] computing brute-force GT ...')
    t0 = time.time()
    gt = []
    for i in range(0, len(queries), 200):
        S = queries[i:i+200] @ base.T
        top = np.argpartition(-S, K, axis=1)[:, :K]
        for r in range(top.shape[0]):
            order = top[r][np.argsort(-S[r][top[r]])]
            gt.append([int(d) for d in order])
    with open(GT, 'wb') as f:
        f.write(struct.pack('>i', len(gt)))
        for g in gt:
            f.write(struct.pack('>i', len(g)))
            f.write(struct.pack('>%di' % len(g), *g))
    print(f'[faiss-laion] GT cached ({time.time()-t0:.0f}s)')

idx = None
t0 = time.time()
IDX = OUT / 'faiss-laion-hnsw.index'
if IDX.exists():
    idx = faiss.read_index(str(IDX))
    if idx.d != dim:
        print(f'[faiss-laion] cached index dim {idx.d} != {dim} -> rebuild')
        idx = None
    else:
        print('[faiss-laion] index loaded from cache')
if IDX.exists() and idx is None:
    IDX.unlink()
if not IDX.exists() or idx is None:
    idx = faiss.IndexHNSWFlat(dim, 16)
    idx.hnsw.efConstruction = 100
    idx.add(base)
    faiss.write_index(idx, str(IDX))
    print(f'[faiss-laion] built in {time.time()-t0:.0f}s, saved')

# warmup
idx.hnsw.efSearch = 1600
for i in range(min(200, len(queries))):
    idx.search(queries[i:i+1], K)

def recall_at(top, g):
    gs = set(g[:K])
    return sum(1 for d in top[0] if d in gs) / K

print('[faiss-laion] efSearch sweep (recall@10 vs engine-space GT):')
rows = []
for ef in [100, 400, 1600, 6400]:
    idx.hnsw.efSearch = ef
    faiss.cvar.hnsw_stats.reset()
    times = []
    recs = []
    for i in range(len(queries)):
        t0q = time.perf_counter_ns()
        D, I = idx.search(queries[i:i+1], K)
        t1q = time.perf_counter_ns()
        times.append(t1q - t0q)
        recs.append(recall_at(I, gt[i]))
    ndis = faiss.cvar.hnsw_stats.ndis / len(queries)
    times.sort()
    p50 = times[len(times)//2] / 1e6
    rows.append((ef, float(np.mean(recs)), ndis, p50))
    print(f'  ef={ef:5d} recall={np.mean(recs):.3f} dc={ndis:.0f} p50={p50:.2f}ms')

with open(OUT / 'faiss-laion.csv', 'w') as f:
    f.write('efSearch,recall10,dcPerQuery,p50ms\n')
    for ef, r, d, p50 in rows:
        f.write(f'{ef},{r:.4f},{d:.0f},{p50:.3f}\n')
print('[faiss-laion] wrote out/faiss-laion.csv')
