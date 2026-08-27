# 本地读取已下载的 lance-format/laion-1m，提取图像嵌入与 caption：
#   out\laion\laion_base.fvecs  — 前 1M 行 img_emb（768 维 float32，fvecs 格式）
#   out\laion\laion_captions.txt — 对应 caption（末 2000 行将作查询集）
import os
import struct
import numpy as np
import lance

DS = r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\laion-full\data\train.lance'
N = 1_000_000
OUT = r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\laion'
os.makedirs(OUT, exist_ok=True)

print('[extract] opening', DS)
ds = lance.dataset(DS)
print('[extract] schema ok, count_rows:', ds.count_rows())

fb = open(os.path.join(OUT, 'laion_base.fvecs'), 'wb')
fc = open(os.path.join(OUT, 'laion_captions.txt'), 'w', encoding='utf-8')
written = 0
try:
    for batch in ds.to_batches(columns=['img_emb', 'caption'], batch_size=10000):
        emb = batch.column('img_emb').to_pylist()      # list of list[float]
        cap = batch.column('caption').to_pylist()
        for i in range(len(emb)):
            if written >= N:
                break
            v = np.asarray(emb[i], dtype=np.float32)
            if v.shape[0] != 768:
                print('[extract] BAD DIM', v.shape, 'at', written)
                continue
            fb.write(struct.pack('<i', 768))
            fb.write(v.tobytes())
            fc.write(str(cap[i]).replace('\n', ' ') + '\n')
            written += 1
        if written >= N:
            break
        print('[extract] written', written, flush=True)
finally:
    fb.close()
    fc.close()
print('[extract] DONE, rows =', written)
