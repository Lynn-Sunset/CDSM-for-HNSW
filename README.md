# CDSM for HNSW

CDSM is a stateful search strategy over an existing HNSW graph. It first searches from the graph's normal entry point, spends a bounded amount of work exploring alternative entries, and then continues from the combined search state. It uses the existing graph without changing its edges.

The alternative searches, called **scouts**, contribute both candidate results and unfinished work. Their discoveries can change which nodes the main search expands next. All routes share a distance cache and node-discovery state, so a node's distance is evaluated at most once per query.

This repository provides method source code and a synthetic usage example. It contains no benchmark datasets, experimental results, or performance claims.

## Search flow

The example uses `paid::search()` in [paid.h](include/cdsm/paid.h). All four demonstrated methods follow the same initial search and share the same state representation.

1. **Initial search.** Descend through HNSW's upper layers and explore the bottom layer in distance order. Pause when the initial distance budget is reached, the stopping rule fires, or no expandable work remains. The stopping rule compares the nearest pending candidate with the current tenth-best result: with squared L2 distances, it uses a factor of `(1 + 0.075)^2`. This is a search rule, not a learned prediction of recall failure.
2. **Optional entry screening.** SCORE and DIVERSE spend part of the remaining budget descending from alternative upper-layer nodes and selecting bottom-layer landing points. C and F skip this phase.
3. **Scouts.** F, SCORE and DIVERSE attempt up to four scouts, each with a cap of 400 additional distance evaluations. A scout explores from its landing point using a fresh local top-10 threshold. Before ten local candidates have been collected, this threshold is infinite; afterwards it is the local tenth-best distance. The threshold controls candidate admission and stopping independently of the global result set.
4. **Merge and continue.** Merge each scout's pending candidates into the main search. Preserve any partially scanned adjacency list. Once scouting finishes, resume the main search up to the remaining total budget, without applying the initial stopping rule.

A scout can stop at its budget, when its active queue becomes empty, or when its nearest active candidate is worse than its local threshold. An empty active queue does **not** necessarily mean all work is exhausted: deferred candidates or partially scanned nodes may still be available for continuation.

### What is preserved?

The implementation is in [cdsm.h](include/cdsm/cdsm.h).

| State | Purpose |
|---|---|
| Shared distance cache | Reuse an already evaluated query-to-node distance across every phase. |
| Global result collector | Maintain the best ten eligible results discovered across all routes. |
| Node flags | Distinguish bottom-layer discovery, entry use, opened adjacency lists and completed expansions. |
| Search frontier | Hold active candidates, deferred candidates and the currently expanding node. |
| Adjacency cursor | Remember where a paused node's neighbor scan should resume. |

After merging, candidates discovered by scouts compete in the main distance-ordered queue. A partially expanded main-search node is finished first. CDSM therefore does more than run independent searches and merge their final top-10 lists: it also transfers unfinished exploration into the continuation.

## Methods and entry selection

| Method in the example | Behavior after the initial search |
|---|---|
| `C` | Directly continue the existing main-search state. This is a control within this implementation, not native Faiss-HNSW search. |
| `F` | Use alternative entries in table order, run scouts, merge their state, then continue. |
| `SCORE` | Pay to preview alternative entries; prefer landing points with smaller query distance, run scouts, then continue. |
| `DIVERSE` | Use the same paid preview; select the first landing point by distance, then favor additional coverage of undiscovered bottom-layer neighbors. Run scouts, then continue. |

`entryTable()` builds a query-independent table of at most 64 distinct nodes that occur above the bottom layer. It traverses upper levels in ascending order and node IDs in ascending order, adding each node once for every upper level it occupies. It samples that pool with a Java-compatible random generator seeded by the graph's node count, rejecting duplicate selections. The table retains sampling order; it is **not sorted by query distance or expected search quality**.

For F, `nextEntry()` takes the first node in this table that has neither been used as a scout entry nor been discovered at the bottom layer. A node evaluated only during upper-layer traversal can still qualify.

SCORE and DIVERSE deterministically shuffle eligible entries using the query's float-array hash, then preview as many as the screening budget permits. Selection uses only completed descents and avoids duplicate landing points. For DIVERSE, additional coverage means the fraction of a landing point's neighbors that were bottom-undiscovered at inspection time and are not covered by earlier selections. Distance and preview order break ties. Unfilled scout slots fall back to table entries.

KEEP and E64_RAW are not included in this source release.

## Budget meaning

The budget counts **new query-to-node distance evaluations**, including upper-layer descent, entry screening, scouts and continuation. Cache hits do not spend another distance evaluation. Neighbor inspections, heap operations and other bookkeeping still take time, so equal distance budgets do not imply equal latency.

`paid::search()` has two budget modes:

- With a nonnegative `absoluteCap`, that value bounds the entire query. The initial cap is also clamped to it; `extra` is ignored.
- Otherwise, the total cap is `P + extra`, where `P` is the initial search's **actual** distance count, which may be smaller than `primaryCap`.

`screenBudget` and each scout's quota are portions of the total budget, not additions to it. A scout's descent is charged to its quota unless a landing point was already obtained through paid screening. The query can finish below its cap if no expandable work remains. Use `run.out.dc` for the actual distance count and `run.cap` for the total cap.

## Build and run

Requirements: a C++17 compiler, CMake 3.18 or newer, OpenMP, and a CPU Faiss installation exposing its `faiss` CMake target. Faiss must be installed separately. The sources have been compiled and the example run with Faiss 1.15.0 and MSVC on Windows.

```sh
git clone https://github.com/Lynn-Sunset/CDSM-for-HNSW.git
cd CDSM-for-HNSW
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/faiss/install
cmake --build build --config Release
```

Replace `/path/to/faiss/install` with the installation prefix, or omit that option if CMake already finds Faiss. With a single-configuration generator, run `./build/cdsm_example`; with a Windows multi-configuration generator, the executable is typically `build/Release/cdsm_example.exe`.

[examples/search.cpp](examples/search.cpp) builds a small synthetic graph and calls C, F, SCORE and DIVERSE. It sets an initial cap of 256, a screening cap of 128 and an absolute total cap of 1024. It checks the budget, result IDs and returned distances. This is a usage and correctness check, not a benchmark.

## Implementation scope

- The current implementation returns up to **10** results (`cdsm::K`) and assumes **squared L2** distances. Do not substitute an inner-product index without adapting the ordering and stopping rules.
- Supply a nonempty compatible `faiss::IndexHNSWFlat`, a query of the index dimension, a mask with one element per indexed vector, valid entry IDs and nonnegative phase budgets. A positive total budget is expected. The API assumes valid input; it is not a fully validated general-purpose interface.
- The mask controls result collection only. It does not prevent traversal through excluded nodes or create a filtering index. The example uses an all-ones mask.
- Create one `Workspace` per concurrent query. It holds mutable search state and arrays proportional to the number of indexed vectors; it can be reused for sequential queries.
- Start reading at `paid::search()`, then follow `Workspace::distanceAdvance()`, `scout()`, `Workspace::advance()` and `score()`. [common.h](include/cdsm/common.h) supplies the required includes and assertion helper.
