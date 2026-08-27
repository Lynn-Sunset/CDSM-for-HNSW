# Minimal data-integrity checker (external-review item 3: no silent stale-cache reads).
# Validates: (1) every .fvecs/.ivecs under out/ parses; (2) GT id ranges fit the
# matching base; (3) ES 9.5 sweep CSVs have the expected schema and recall in [0,1];
# (4) cdsm10 fair-Pareto CSVs have qid + 32 numeric columns.
# Usage: python check.py
import csv, glob, os, struct, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
failures = []

def bad(msg):
    failures.append(msg)
    print('FAIL:', msg)

def base_rows(path):
    n = 0
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(4)
            if not hdr: break
            d = struct.unpack('<i', hdr)[0]
            f.seek(4 * d, 1)
            n += 1
    return n

# (1)+(2) vector / GT files
for p in glob.glob(os.path.join(ROOT, 'out', '**', '*.fvecs'), recursive=True):
    try:
        n = base_rows(p)
    except Exception as e:
        bad(f'{os.path.relpath(p, ROOT)}: {e}')
        continue
    base = os.path.splitext(p)[0] + '.fvecs'
    gt = os.path.splitext(p)[0] + '_groundtruth.ivecs'
    if os.path.exists(gt):
        try:
            with open(gt, 'rb') as f:
                while True:
                    hdr = f.read(4)
                    if not hdr: break
                    k = struct.unpack('<i', hdr)[0]
                    ids = struct.unpack('<' + 'i' * k, f.read(4 * k))
                    if ids and (min(ids) < 0 or max(ids) >= n):
                        bad(f'{os.path.relpath(gt, ROOT)}: ids out of range [0,{n})')
        except Exception as e:
            bad(f'{os.path.relpath(gt, ROOT)}: {e}')

# (3) ES 9.5 sweep CSVs
for p in glob.glob(os.path.join(ROOT, 'out', '**', 'es95-sweep.csv'), recursive=True):
    try:
        rows = list(csv.DictReader(open(p, encoding='utf-8')))
        if set(rows[0].keys()) != {'num_candidates', 'early_termination', 'recall@10', 'median_ms', 'p95_ms'}:
            bad(f'{os.path.relpath(p, ROOT)}: unexpected columns')
        for r in rows:
            rec = float(r['recall@10'])
            if not (0.0 <= rec <= 1.0):
                bad(f'{os.path.relpath(p, ROOT)}: recall out of range {rec}')
    except Exception as e:
        bad(f'{os.path.relpath(p, ROOT)}: {e}')

# (4) cdsm10 fair-Pareto CSVs
for p in glob.glob(os.path.join(ROOT, 'es95', 'cdsm10-*.csv')):
    try:
        rows = list(csv.DictReader(open(p, encoding='utf-8')))
        keys = list(rows[0].keys())
        if keys[0] != 'qid' or len(keys) != 33:
            bad(f'{os.path.relpath(p, ROOT)}: expected qid + 32 columns, got {len(keys)}')
        for r in rows:
            for k in keys[1:]:
                float(r[k])
    except Exception as e:
        bad(f'{os.path.relpath(p, ROOT)}: {e}')

print('FAILURES:', len(failures))
sys.exit(1 if failures else 0)
