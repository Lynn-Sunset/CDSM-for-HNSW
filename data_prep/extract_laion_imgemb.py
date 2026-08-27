# LAION-1M shard -> real CLIP image embeddings (openai/clip-vit-base-patch32 vision tower).
# Chunked: read a row-chunk from Lance, close reader, then decode+encode (avoids thread contention).
# Resumable: counts rows already in laion_img_base.fvecs and skips that many shard rows.
# Outputs (append): out/laion/laion_img_base.fvecs (unit-norm float32 512-d),
#                   out/laion/laion_img_captions.txt, out/laion/laion_img_sim.txt
import os, sys, io, time, struct
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np, torch, torchvision
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizer
from lance.file import LanceFileReader
from PIL import Image

WS = r'C:\Users\liuruilin\Desktop\es\hnsw-phase0'
SHARD = WS + r'\out\laion-full\data\train.lance\data\011110001010001111001111d118ca4cb48ab71d105ee40181.lance'
OB = WS + r'\out\laion\laion_img_base.fvecs'
OC = WS + r'\out\laion\laion_img_captions.txt'
OS_ = WS + r'\out\laion\laion_img_sim.txt'
DIM = 512
ROWBYTES = 4 + DIM * 4
CHUNK = 8192
BATCH = 64
MAXN = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9

torch.set_num_threads(12)
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
model.eval()
tf = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                         transforms.CenterCrop(224), transforms.ConvertImageDtype(torch.float32),
                         transforms.Lambda(lambda t: (t[:3] if t.shape[0] >= 3 else t.repeat(3, 1, 1)[:3])),
                         transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

def decode(b):
    t = None
    try:
        t = torchvision.io.decode_image(torch.frombuffer(bytearray(b), dtype=torch.uint8))
    except Exception:
        try:
            t = transforms.functional.pil_to_tensor(Image.open(io.BytesIO(b)).convert('RGB'))
        except Exception:
            return None
    if t is not None and t.dim() == 4:
        t = t[0]  # animated GIF: take first frame
    return t

done = 0
if os.path.exists(OB):
    done = os.path.getsize(OB) // ROWBYTES
print('[imgemb] resume: %d rows already written' % done, flush=True)

reader = LanceFileReader(SHARD)
total = reader.metadata().num_rows
print('[imgemb] shard rows =', total, flush=True)
pos = done
written = 0; skipped = 0
t_start = time.time()
fb = open(OB, 'ab'); fc = open(OC, 'a', encoding='utf-8'); fs = open(OS_, 'a', encoding='utf-8')
val_caps = []; val_sims = []; val_embs = []
try:
    while pos < total and written < MAXN:
        n = min(CHUNK, total - pos)
        rows = []
        for batch in reader.read_range(pos, n, batch_size=1024).to_batches():
            imgs = batch.column('image'); caps = batch.column('caption'); sims = batch.column('similarity')
            for j in range(len(imgs)):
                cap = caps[j].as_py()
                b = imgs[j].as_py()
                sim = sims[j].as_py()
                if not isinstance(cap, str) or b is None or len(b) == 0 or sim is None:
                    skipped += 1; continue
                rows.append((b, cap, sim))
        pos += n
        buf = []
        for (b, cap, sim) in rows:
            t = decode(b)
            if t is None:
                skipped += 1; continue
            xt = tf(t)
            if xt.dim() != 3 or xt.shape[0] != 3:
                skipped += 1; continue
            buf.append((xt, cap, sim))
            if len(buf) == BATCH:
                x = torch.stack([e for e, _, _ in buf])
                with torch.no_grad():
                    e = model.get_image_features(pixel_values=x)
                    if hasattr(e, 'pooler_output'): e = e.pooler_output
                    e = torch.nn.functional.normalize(e, dim=-1)
                en = e.numpy().astype(np.float32)
                for k in range(len(buf)):
                    fb.write(struct.pack('<i', DIM)); fb.write(en[k].tobytes())
                    fc.write(buf[k][1].replace('\n', ' ') + '\n')
                    fs.write('%.6f\n' % buf[k][2])
                    if len(val_caps) < 1024 and (written + k) % 97 == 0:
                        val_embs.append(en[k]); val_caps.append(buf[k][1]); val_sims.append(buf[k][2])
                written += len(buf); buf = []
                if written >= MAXN: break
        if buf:
            x = torch.stack([e for e, _, _ in buf])
            with torch.no_grad():
                e = model.get_image_features(pixel_values=x)
                if hasattr(e, 'pooler_output'): e = e.pooler_output
                e = torch.nn.functional.normalize(e, dim=-1)
            en = e.numpy().astype(np.float32)
            for k in range(len(buf)):
                fb.write(struct.pack('<i', DIM)); fb.write(en[k].tobytes())
                fc.write(buf[k][1].replace('\n', ' ') + '\n')
                fs.write('%.6f\n' % buf[k][2])
            written += len(buf)
        fb.flush(); fc.flush(); fs.flush()
        el = time.time() - t_start
        rate = written / max(el, 1e-6)
        print('[imgemb] pos=%d written=%d skipped=%d rate=%.3f img/s ETA=%.0fmin' %
              (pos, written, skipped, rate, (min(total - pos, MAXN - written)) / max(rate, 1e-6) / 60.0), flush=True)
finally:
    fb.close(); fc.close(); fs.close()
print('[imgemb] DONE written=%d skipped=%d in %.0fmin' % (written, skipped, (time.time() - t_start) / 60.0), flush=True)

if val_caps:
    tok = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    E = np.vstack(val_embs)
    with torch.no_grad():
        t = tok(val_caps, padding=True, truncation=True, return_tensors='pt')
        te = model.get_text_features(**t)
        if hasattr(te, 'pooler_output'): te = te.pooler_output
        te = torch.nn.functional.normalize(te, dim=-1).numpy()
    ct = (E * te).sum(-1)
    s = np.asarray(val_sims, dtype=np.float64)
    r = np.corrcoef(ct, s)[0, 1]
    print('[imgemb] VALIDATION n=%d my_cos=%.3f+-%.3f dataset_sim=%.3f+-%.3f Pearson r=%.3f' %
          (len(ct), ct.mean(), ct.std(), s.mean(), s.std(), r), flush=True)
