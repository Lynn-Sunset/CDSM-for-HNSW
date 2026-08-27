# GIST1M (open-vdb parquet) -> texmex fvecs/ivecs, matching PhaseSIFT loaders:
#   fvecs: per vector: big-endian int dim + little-endian float32[]
#   ivecs: per row:   big-endian int k + big-endian int32[]
import struct
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r'C:\Users\liuruilin\.cache\huggingface\hub\datasets--open-vdb--gist-960-euclidean\snapshots\9230dd86c71f7b421a15110495ec6b98bd64981e')
OUT = Path(r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\gist')

OUT.mkdir(parents=True, exist_ok=True)

def load_train():
    frames = [pd.read_parquet(BASE / 'train' / f'train-0000{i}-of-00003.parquet') for i in (1, 2, 3)]
    df = pd.concat(frames, ignore_index=True)
    X = np.stack([np.asarray(v, dtype=np.float32) for v in df['emb']])
    print(f'train: {X.shape[0]} x {X.shape[1]}')
    return X

def write_fvecs(path, X):
    with open(path, 'wb') as f:
        for i in range(X.shape[0]):
            f.write(struct.pack('<i', X.shape[1]))   # little-endian dim (texmex convention)
            f.write(X[i].tobytes())                  # little-endian float32
    print(f'wrote {path} ({X.shape[0]} vectors)')

X = load_train()
write_fvecs(OUT / 'gist_base.fvecs', X)
del X

te = pd.read_parquet(BASE / 'test' / 'test-00001-of-00001.parquet')
Q = np.stack([np.asarray(v, dtype=np.float32) for v in te['emb']])
print(f'test: {Q.shape[0]} x {Q.shape[1]}')
write_fvecs(OUT / 'gist_query.fvecs', Q)

nb = pd.read_parquet(BASE / 'neighbors' / 'neighbors-vector-emb-pk-idx-expr-None-metric-l2.parquet')
ids = nb['neighbors_id'].tolist()
dists = nb['neighbors_distance'].tolist()
# sanity: distances ascending, ids in range
bad = sum(1 for d in dists if list(d) != sorted(d))
print(f'neighbors rows: {len(ids)}, unsorted-distance rows: {bad}')
with open(OUT / 'gist_groundtruth.ivecs', 'wb') as f:
    for row in ids:
        row100 = [int(x) for x in row[:100]]
        f.write(struct.pack('<i', len(row100)))
        f.write(struct.pack(f'<{len(row100)}i', *row100))
print(f'wrote {OUT / "gist_groundtruth.ivecs"} ({len(ids)} rows)')
print('conversion done')
