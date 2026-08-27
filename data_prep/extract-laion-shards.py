# 从已下载的 Lance 分片提取 img_emb + caption（用 LanceFileReader 单文件直读，
# 不需要数据集 manifest）：
#   out\laion\laion_base.fvecs    — 全部可读行的 768 维 float32 嵌入
#   out\laion\laion_captions.txt  — 对应 caption
import glob
import os
import struct
import numpy as np
from lance.file import LanceFileReader

ROOT = r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\laion-full'
OUT = r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\laion'
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(os.path.join(ROOT, 'data', 'train.lance', 'data', '*.lance')))
print('[extract] shards:', [os.path.basename(f) for f in files])

fb = open(os.path.join(OUT, 'laion_base.fvecs'), 'wb')
fc = open(os.path.join(OUT, 'laion_captions.txt'), 'w', encoding='utf-8')
written = 0
try:
    for f in files:
        print('[extract] reading', os.path.basename(f), flush=True)
        reader = LanceFileReader(str(f))
        for batch in reader.read_all(batch_size=10000).to_batches():
            emb = batch.column('img_emb').to_pylist()
            cap = batch.column('caption').to_pylist()
            for j in range(len(emb)):
                v = np.asarray(emb[j], dtype=np.float32)
                if v.shape[0] != 768:
                    print('[extract] BAD DIM', v.shape, 'shard', os.path.basename(f), 'row', j)
                    continue
                fb.write(struct.pack('<i', 768))
                fb.write(v.tobytes())
                fc.write(str(cap[j]).replace('\n', ' ') + '\n')
                written += 1
        print('[extract] cumulative rows:', written, flush=True)
finally:
    fb.close()
    fc.close()
print('[extract] DONE, rows =', written)
