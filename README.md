# Code artifact for "The Gate Is the Gap: Budget Recovery for Gated HNSW Engines via CDSM"

Structured packaging of all code the paper relies on. Data (out/*.csv, indexes) and
figures are outputs, not packaged; every paper number traces to these runners.

## Layout
- java/            bytecode-faithful 9.12 port + all phase runners (compile against lucene-core-9.12.0)
- faiss/           faiss baselines and the faiss transplant (negative result 1)
- data_prep/       corpus ingestion / LAION image-embedding recompute (provenance, S2.2)
- figs_gen/        figure generators (Figs 1-4, Fig 3 panels)
- run_scripts/     PowerShell drivers used for the reported sweeps
- analysis/        ground-truth / sanity checkers
- evidence/        javap -c disassembly snapshots: bytecode_9_12 (S2.1 port verification),
                   bytecode_10_5 (S6.9 ES 9.5.2 / Lucene 10.5.1 verification)

## Paper -> code map
| Paper item | Code |
|---|---|
| S2.1 bytecode-faithful port (two predicates, score-before-gate) | java/InstrumentedSearch.java, java/InstrumentedSimSearch.java |
| S2.1 port == 9.12 bytecode | evidence/bytecode_9_12/HnswGraphSearcher.disasm.txt, NeighborQueue.disasm.txt |
| S2.4/S3.3 kernel microbenchmark (Panama vs scalar vs faiss) | java/LuceneKernelBench.java |
| S3.1/S6.3/S6.7 GIST port sweeps, CDSM family configs | java/Phase0Main.java, java/PhaseGISTLatency.java, java/PhaseGISTEndToEnd.java |
| S3.2/S6.1/S6.2 T2I real-engine end-to-end (gate intact vs searcher-patch deleted) | java/PhaseT2IEndToEnd.java, java/PhaseT2ILatency.java |
| S6.1/S6.2 LAION-CLIP real engine | java/PhaseLAIONEndToEnd.java, java/PhaseLAIONIndex.java |
| S6.4 SIFT + segmented production shape | java/PhaseSIFT.java, java/PhaseSIFTLatency.java, java/PhaseSIFTSegLatency.java, faiss/faiss-seg-sift*.py |
| S6.5 tail rescue | java/PhaseGISTLatency.java (catastrophic slicing), faiss/faiss-seg-sift2.py |
| S6.7(1) faiss transplant (multi-entry on gate-free engine) | faiss/faiss-cdsm.py |
| S6.7(5) eps[] proxy (shared visited + shared budget multi-seed) | java/InstrumentedSearch.java searchTwoSeed() |
| S4/Fig 3 real-data panels (SIFT q2130 anatomy) | java/PhaseRealVis.java + figs_gen/make-realvis-figs.py |
| Fig 3 schematic panels | figs_gen/make-theory-figs.py |
| Figs 1/2/4 and theory data figs | figs_gen/make-figs.py |
| S5 design-space sweeps (entry forms Lite/Far/Base, marks, budgets) | java/PhaseAEntry.java, java/PhaseAFair.java, java/PhaseAHetero.java, java/PhaseASplit.java, java/Phase1*.java |
| S6.9 ES 9.5.2 / Lucene 10.5.1 verification | evidence/bytecode_10_5/*.disasm.txt (REST sweep protocol in supplemental) |
| S2.2 corpora ingestion; LAION provenance (img_emb mislabel recompute) | data_prep/* |

## Dependencies
- Java 17+ with lucene-core 9.12.0 on the classpath (port and runners)
- Python: faiss-cpu 1.14.3, numpy, pandas, matplotlib, networkx (figs), torch+transformers+PIL (LAION recompute), pylance (LAION shard read)
See requirements.txt for the Python side.
