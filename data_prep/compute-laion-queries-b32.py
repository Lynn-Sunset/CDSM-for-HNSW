# LAION-CLIP slice: text-side queries = openai/clip-vit-base-patch32 TEXT embeddings
# of the last 2000 captions in out/laion/laion_captions.txt (unit-norm, 512-d).
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import struct
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

WS = r'C:\Users\liuruilin\Desktop\es\hnsw-phase0'
CAPS = WS + r'\out\laion\laion_captions.txt'
OUTQ = WS + r'\out\laion\laion_query.fvecs'
NQ = 2000

caps = open(CAPS, encoding='utf-8').read().splitlines()
print('[q-b32] captions', len(caps))
query_caps = caps[-NQ:]

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
tok = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
model.eval()
embs = []
with torch.no_grad():
    for i in range(0, NQ, 128):
        batch = query_caps[i:i+128]
        t = tok(batch, padding=True, truncation=True, return_tensors='pt')
        te = model.get_text_features(**t)
        if hasattr(te, 'pooler_output'):
            te = te.pooler_output
        te = torch.nn.functional.normalize(te, dim=-1)
        embs.append(te.numpy())
emb = np.vstack(embs).astype(np.float32)
with open(OUTQ, 'wb') as f:
    for i in range(NQ):
        f.write(struct.pack('<i', emb.shape[1]))
        f.write(emb[i].tobytes())
print('[q-b32] wrote', OUTQ, emb.shape)
