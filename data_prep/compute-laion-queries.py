# 查询集构造：验证 CLIP 文本编码器与 img_emb 空间对齐（候选比对），
# 用对齐者编码末 2000 条 caption → out\laion\laion_query.fvecs。
import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import struct
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

BASE = r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\laion\laion_base.fvecs'
CAPS = r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\laion\laion_captions.txt'
OUTQ = r'C:\Users\liuruilin\Desktop\es\hnsw-phase0\out\laion\laion_query.fvecs'
NQ = 2000

def load_fvecs(p):
    with open(p, 'rb') as f:
        xs = []
        while True:
            hdr = f.read(4)
            if not hdr:
                break
            dim = struct.unpack('<i', hdr)[0]
            xs.append(np.frombuffer(f.read(4 * dim), dtype=np.float32))
    return np.vstack(xs)

base = load_fvecs(BASE)
print('[q] base', base.shape)
caps = open(CAPS, encoding='utf-8').read().splitlines()
print('[q] captions', len(caps))

cands = ['openai/clip-vit-large-patch14', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K']
chosen = None
for m in cands:
    try:
        model = CLIPModel.from_pretrained(m)
        tok = CLIPTokenizer.from_pretrained(m)
        model.eval()
        idxs = [1000, 1001, 1002, 1003, 1004]
        with torch.no_grad():
            t = tok([caps[i] for i in idxs], padding=True, truncation=True, return_tensors='pt')
            te = model.get_text_features(**t)
            if hasattr(te, 'pooler_output'):
                te = te.pooler_output
            te = torch.nn.functional.normalize(te, dim=-1)
            ie = base[idxs] / np.linalg.norm(base[idxs], axis=1, keepdims=True)
            sims = (te.numpy() * ie).sum(-1)
        print('[q]', m, 'caption-img cosine:', np.round(sims, 3).tolist(), 'mean', round(float(sims.mean()), 3))
        if chosen is None or sims.mean() > chosen[1]:
            chosen = (m, float(sims.mean()), model, tok)
    except Exception as e:
        print('[q]', m, 'FAILED', type(e).__name__, str(e)[:150])

if chosen is None or chosen[1] < 0.10:
    print('[q] NO model aligns with img_emb space (mean sim < 0.10) — abort')
    raise SystemExit(1)

print('[q] chosen model:', chosen[0], 'mean sim', chosen[1])
model, tok = chosen[2], chosen[3]
query_caps = caps[-NQ:]
embs = []
with torch.no_grad():
    for i in range(0, NQ, 64):
        batch = query_caps[i:i+64]
        t = tok(batch, padding=True, truncation=True, return_tensors='pt')
        te = model.get_text_features(**t)
        if hasattr(te, 'pooler_output'):
            te = te.pooler_output
        te = te / te.norm(dim=-1, keepdim=True)
        embs.append(te.numpy())
emb = np.vstack(embs).astype(np.float32)
with open(OUTQ, 'wb') as f:
    for i in range(NQ):
        f.write(struct.pack('<i', 768))
        f.write(emb[i].tobytes())
print('[q] wrote', OUTQ, emb.shape)
