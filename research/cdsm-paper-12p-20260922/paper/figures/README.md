# Reproducible paper figures

The `.tex` and external `.csv` files are authoritative editable sources. PDFs are vector outputs; `.preview.png` files are only for visual inspection. No ANN experiment is run by the figure script.

The revised paper has four figures: `method-feedback` (conceptual execution organization), `t2i-budget` (fixed work), `native-tradeoff` (190–340 QPS zoom), and `native-full` (complete range of the same native data). The method diagram is authored directly in TeX; the other three are generated from CSV. No diagram node or arrow represents a measured trajectory. Use the package-level `build.ps1 -RegenerateAssets` to rebuild all four.

The generator first reads the packaged frozen summaries under `../../source-data/`, resolving this path relative to the script rather than the current working directory. The expected paths under that directory are:

- `fanng-search-comparison-20260922/SUMMARY.json`
- `hnsw-adaptive-baselines-20260920/server-results/final-20260921-005721/results/SUMMARY.json`

If a packaged summary is absent, the generator falls back to the same research-relative path three directories above `figures`, supporting the original workspace layout. A complete package needs no original workstation paths. `PROVENANCE.json` records the logical research-relative source paths and the hashes of the actual files read, regardless of which location is used.

From this directory:

```powershell
python generate_figures.py
pdflatex -interaction=nonstopmode -halt-on-error t2i-budget.tex
pdflatex -interaction=nonstopmode -halt-on-error t2i-budget.tex
pdflatex -interaction=nonstopmode -halt-on-error native-tradeoff.tex
pdflatex -interaction=nonstopmode -halt-on-error native-tradeoff.tex
pdftoppm -png -r 200 -singlefile t2i-budget.pdf t2i-budget.preview
pdftoppm -png -r 200 -singlefile native-tradeoff.pdf native-tradeoff.preview
```

The two source-summary hashes and every extracted CSV hash are in `PROVENANCE.json`. All CSV values are extracted from frozen summaries, not transcribed from manuscript tables. Lines connect observed settings; no fitted values, confidence bands, or interpolation claims are introduced.

Suggested captions:

**t2i-budget.pdf:** Recall and severe failures across all three fixed actual-distance budgets on four T2I graphs (8,000 queries per graph). Every method spends exactly the plotted number of distance evaluations on each query. FANNG denotes the same-HNSW-graph search reconstruction, not the original FANNG index system. Lines connect the measured settings; repeated query identities across graphs are not independent samples.

**native-tradeoff.pdf / native-full.pdf:** Enlarged high-recall region and full-range quality--throughput tradeoffs on Faiss-M32 and Imported-s777. Both use identical CSV data with all allowed measured points, including dominated ones. All C, F, native HNSW, uncapped ABS with gamma below 0.06, and capped ABS values come from the earlier timing wave; no QPS is imported from the later FANNG run. ABS_CAP is unconnected because both cap and gamma vary. These are quality--cost curves, not matched per-query work. All four graphs' complete summaries remain in the package.

Severe means Recall@10 < 0.5. The full native evaluation grid retained here is the available frozen evaluation grid, not every development-grid parameter. QPS is not converted into a new experimental estimate.
