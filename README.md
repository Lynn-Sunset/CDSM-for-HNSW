# CDSM: Budgeted Multi-Entry Search for Reducing Severe Recall Failures

CDSM gives alternative entrances bounded local searches (**scouts**), then
returns their unfinished state to the main frontier. This can change which nodes
the main search expands next. Scouts and the main search share distance caches,
visited state, adjacency-scan progress, and one query budget. Entrance scoring
and upper-layer descent are charged.

This repository contains the current 12-page manuscript, its frozen evidence,
and server experiment sources. Authors and publication metadata remain blank;
the manuscript is a draft.

## Current manuscript and results — 2026-09-23

- [Manuscript PDF: 12 pages including appendix and references](research/cdsm-paper-12p-20260922/output/pdf/cdsm-paper.pdf)
- [Editable manuscript source archive](research/cdsm-paper-12p-20260922/output/cdsm-paper-source.zip)
- [Latest revision report, definitions and comparisons (Chinese)](research/cdsm-paper-12p-20260922/R2-REVISION-RESPONSE.zh-CN.md)
- [Paper build and evidence guide (Chinese)](research/cdsm-paper-12p-20260922/README.zh-CN.md)
- [Server protocol](research/cdsm-paper-12p-20260922/source-data/cdsm-review-strengthening-20260922/PROTOCOL.zh-CN.md) and [runbook](research/cdsm-paper-12p-20260922/source-data/cdsm-review-strengthening-20260922/RUNBOOK.zh-CN.md)
- [Results](research/cdsm-paper-12p-20260922/source-data/cdsm-review-strengthening-20260922/SUMMARY.json), [server audit](research/cdsm-paper-12p-20260922/source-data/cdsm-review-strengthening-20260922/AUDIT.json), and [local verification](research/cdsm-paper-12p-20260922/source-data/cdsm-review-strengthening-20260922/LOCAL-VERIFICATION.json)

The latest supplement adds shared-queue multi-entry search with 1/4/8/16
entrances (**MEP**), development-selected MEP, scouts rooted in the existing main
frontier (**FRONTIER**), and neighboring scout quotas. Original CDSM **F** remains
frozen at up to four scouts with a 400-distance quota each.

The following ranges cover four fixed T2I graphs: Faiss M32 and imported Lucene
M16 graphs with seeds 42, 777, and 1234. Each graph uses the same 8,000 evaluation
queries and **exactly 10,240 actual distance computations per query** in this
comparison. **C** spends the remaining budget continuing the original search in
the same executor; it is not the separate native Faiss baseline.

| Policy versus C | Mean Recall@10 gain, percentage points | Relative reduction in severe failures | QPS change |
|---|---:|---:|---:|
| F200: up to 200 distances per scout | +0.188 to +0.309 | 41% to 66% | -0.77% to -0.07% |
| Original F: up to 400 distances per scout | +0.214 to +0.398 | 44% to 72% | -3.28% to -2.58% |

A severe failure retrieves only 0–4 of the true top-10 neighbors. The 1.5% QPS
tolerance is a practical reporting convention, not a statistical equivalence
test. QPS is the inverse of mean per-query median latency over five timed rounds
on 800 fixed evaluation queries. F200 is a predeclared depth check, not a
replacement selected for the frozen original F.

At the primary budget, F has higher mean recall than all four MEP sizes on all
four graphs. Against development-selected MEP, F costs 3.70%–7.28% QPS; the severe
failure difference survives the prespecified Holm correction on only one graph.
MEP-16 has fewer severe failures than F on two graphs. FRONTIER is a custom
control, not native iQAN, and its candidate-set construction affects its timing.
The full report retains these tradeoffs and the other budget points.

The supplement contains 92,000 development, 736,000 evaluation, and 368,000 timing
records. These are repeated searches, not independent query counts. Development
and evaluation identities are disjoint, but both were used in earlier work: this
is a frozen-policy comparison, not fresh-query confirmation. The manuscript also
retains the F → SCORE → DIVERSE → KEEP → E64_RAW series and negative transfer on
LAION and WebVid. It does not claim universal gains across datasets or budgets.

## Repository layout

| Path | Contents |
|---|---|
| `research/cdsm-paper-12p-20260922/paper/` | LaTeX manuscript, bibliography, vector figures, and tables |
| `research/cdsm-paper-12p-20260922/source-data/` | 110 frozen source/evidence files and provenance |
| `research/cdsm-paper-12p-20260922/source-data/cdsm-review-strengthening-20260922/` | Latest C++ executor and controls, analysis, query IDs, protocol, selection freeze, results, and audits |
| `tools/verify_artifact.py` | Standard-library verification of package and evidence hashes |

## Rebuild the paper and verify the artifact

From the repository root, using Python 3:

```text
python tools/verify_artifact.py
```

To build the paper, run PowerShell in `research/cdsm-paper-12p-20260922/`:

```powershell
.\build.ps1
# Optional: regenerate tables and figures from the bundled evidence.
.\build.ps1 -RegenerateAssets -Python python
```

Building needs `pdflatex`, `bibtex`, and the required LaTeX packages. Figure
regeneration also needs TikZ/PGFPlots and `standalone`. These commands do not run
ANN. The delivered PDF, manuscript/figure sources, and all 110 frozen evidence
files retain their reviewed bytes. The source archive was repackaged after
removing obsolete notes; its current hash is checked by `verify_artifact.py`.
Git attributes preserve evidence bytes across operating systems.

## Reproduce the server experiments

Start with the [runbook](research/cdsm-paper-12p-20260922/source-data/cdsm-review-strengthening-20260922/RUNBOOK.zh-CN.md).
The experiment sources archive the measured implementation; they are not a
self-contained installation. Full replay also needs the listed vectors, queries,
indexes, ground truth, Faiss library, compiler, Python environment, and historical
parity-check records. Input identities and paths are in the plan and manifests.

Use a new server directory for replay. The archived `compile.sh` contains the
original absolute working directory: update that line in the new copy before
building, then create new freeze records. Do not overwrite the original run or
treat copied completion receipts as a new run. The common `resource_runner.py`
dependency is included under the neighboring source-data directory.

Large vectors/indexes and the 222,842,663-byte raw server archive are retained
separately, not committed to Git. Its [manifest](research/cdsm-paper-12p-20260922/source-data/cdsm-review-strengthening-20260922/SERVER-BUNDLE.json)
records all 1,818 archived files. Raw archive SHA-256:

```text
268f08847ce4fec5890d9a19bcb9491fb55f0d4eb3b4026992100565ceacfb4d
```

The earlier gate-focused Java/Faiss artifact and superseded review plans are
available in [Git history](https://github.com/Lynn-Sunset/CDSM-for-HNSW/tree/63de357f0334ddcac4b2692df6272a5b732241da).
Current build and replay dependencies are documented above and in the runbook.

Special thanks to 上海深至信息科技有限公司，especially engineer Chen Jiuxu.
