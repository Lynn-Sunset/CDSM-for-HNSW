package phase0;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.hnsw.HnswGraph;

import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * Similarity-space port of Lucene 9.12 HnswGraphSearcher, bytecode-faithful,
 * for real production graphs read from Lucene segments.
 *
 * <p>Score semantics: HIGHER = better (cosine dot product). The candidate
 * frontier is a best-first heap gated by minAcceptedSimilarity (worst
 * collected result, tightened dynamically); {@code visitLimit} mirrors the
 * collector visited-count budget (ES num_candidates analog).
 */
public final class InstrumentedSimSearch {

  @FunctionalInterface
  public interface ScorerFn {
    float score(int ord);
  }

  public static final class Result {
    public int[] topK = new int[0];
    public float[] topKScores = new float[0];
    public long scoreComps;        // total score() calls (descent + base layer)
    public long lastImproveAt;     // scoreComps value at last top-k set improvement
    public long bestImproveAt;     // scoreComps value at last best-score improvement
    public int numVisited;
    public int numExpanded;
    public int entryNode;
    public int descendedNode;
    public boolean earlyTerminated;
    /** Visited set at the end of the search (for path-marking / warm-start variants). */
    public BitSet visited;
    /** scoreComps value at every top-k set improvement (trace for signal analysis). */
    public int[] improveTrace = new int[0];
    /** Frontier (unexpanded candidates) captured at gate-closure time — the
     *  boundary of the threshold subgraph. Null unless the gate closed. */
    public int[] frontier;
    public float[] frontierScores;

    public long patience() {
      return scoreComps - lastImproveAt;
    }

    public double patienceRatio() {
      return scoreComps == 0 ? 0.0 : (double) patience() / scoreComps;
    }

    public float bestScore() {
      return topKScores.length == 0 ? Float.NEGATIVE_INFINITY : topKScores[0];
    }
  }

  public static Result search(HnswGraph graph, int size, ScorerFn scorer, int k,
                              int visitLimit) throws IOException {
    return search(graph, size, scorer, k, visitLimit, null, graph.entryNode(), new int[0]);
  }

  /** With acceptOrds (live docs): non-accepted nodes may still bridge the
   *  traversal but never enter the result set (Lucene semantics). */
  public static Result search(HnswGraph graph, int size, ScorerFn scorer, int k,
                              int visitLimit, org.apache.lucene.util.Bits acceptOrds)
      throws IOException {
    return search(graph, size, scorer, k, visitLimit, acceptOrds, graph.entryNode(), new int[0]);
  }

  /** Descent starting from {@code startOrd} (restart oracle). */
  public static Result search(HnswGraph graph, int size, ScorerFn scorer, int k,
                              int visitLimit, int startOrd) throws IOException {
    return search(graph, size, scorer, k, visitLimit, null, startOrd, new int[0]);
  }

  /** Descent starting from {@code startOrd} with acceptOrds (restart oracle). */
  public static Result search(HnswGraph graph, int size, ScorerFn scorer, int k,
                              int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                              int startOrd) throws IOException {
    return search(graph, size, scorer, k, visitLimit, acceptOrds, startOrd, new int[0]);
  }

  /** Two-seed variant: descent from {@code seedA}, beam seeded with
   *  {descendedNode, seedB} (Lucene eps[] mechanism). */
  public static Result searchTwoSeed(HnswGraph graph, int size, ScorerFn scorer, int k,
                                     int visitLimit, int seedA, int seedB) throws IOException {
    return search(graph, size, scorer, k, visitLimit, null, seedA, new int[]{seedB});
  }

  /** Two-seed variant with acceptOrds. */
  public static Result searchTwoSeed(HnswGraph graph, int size, ScorerFn scorer, int k,
                                     int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                                     int seedA, int seedB) throws IOException {
    return search(graph, size, scorer, k, visitLimit, acceptOrds, seedA, new int[]{seedB});
  }

  /** Multi-seed variant: normal descent from {@code primaryStart}, then beam search
   *  seeded with {descendedNode} ∪ {@code extraSeeds} (Lucene eps[] mechanism).
   *  Extra seeds do not descend; they enter the base-layer beam directly. */
  public static Result searchMultiSeed(HnswGraph graph, int size, ScorerFn scorer, int k,
                                       int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                                       int primaryStart, int[] extraSeeds) throws IOException {
    return search(graph, size, scorer, k, visitLimit, acceptOrds, primaryStart, extraSeeds);
  }

  /** Idea-3 variant: beam seeded with the descended node PLUS the best
   *  {@code numExtraSeeds} distinct nodes that were ALREADY scored during the
   *  descent (zero extra distance computations for the extra seeds — they were
   *  paid for by the greedy walk). Local seeds share the basin and the visited
   *  set, so no budget dilution. */
  public static Result searchLocalMultiSeed(HnswGraph graph, int size, ScorerFn scorer, int k,
                                            int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                                            int numExtraSeeds) throws IOException {
    Result r = new Result();
    long dc = 0;
    long budget = visitLimit > 0 ? visitLimit : Long.MAX_VALUE;
    BitSet visited = new BitSet(size);
    MinHeapS worst = new MinHeapS(k + 1);
    int expanded = 0;

    // ---- descent, tracking the top (numExtraSeeds+1) scored nodes ----
    int eps = graph.entryNode();
    r.entryNode = eps;
    float bestScore = scorer.score(eps);
    dc++;
    r.bestImproveAt = dc;
    MinHeapS topN = new MinHeapS(numExtraSeeds + 2);
    topN.add(eps, bestScore);
    int topLevel = graph.numLevels() - 1;

    for (int level = topLevel; level >= 1 && dc < budget; level--) {
      boolean changed = true;
      visited.set(eps);
      while (changed) {
        changed = false;
        graph.seek(level, eps);
        int n;
        while ((n = graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
          if (visited.get(n)) {
            continue;
          }
          visited.set(n);
          if (dc >= budget) {
            r.earlyTerminated = true;
            return finish(r, dc, visited, worst, expanded);
          }
          float s = scorer.score(n);
          dc++;
          topN.add(n, s);
          if (topN.size > numExtraSeeds + 1) {
            topN.poll();
          }
          if (s > bestScore) {
            bestScore = s;
            eps = n;
            changed = true;
            r.bestImproveAt = dc;
          }
        }
      }
    }
    r.descendedNode = eps;

    // seeds = {eps} ∪ {top distinct descent-scored nodes excluding eps}
    List<int[]> sn = new ArrayList<>();
    while (topN.size > 0) {
      sn.add(new int[]{topN.ords[0], Float.floatToIntBits(topN.scores[0])});
      topN.poll();
    }
    java.util.Collections.reverse(sn);   // best first
    int[] seeds = new int[1 + numExtraSeeds];
    seeds[0] = eps;
    int idx = 1;
    for (int[] e : sn) {
      if (e[0] == eps) {
        continue;
      }
      seeds[idx++] = e[0];
      if (idx == seeds.length) {
        break;
      }
    }
    while (idx < seeds.length) {
      seeds[idx++] = eps;   // duplicates are skipped by the visited check
    }

    // ---- beam (identical to the standard searchLevel) ----
    visited.clear();
    MaxHeapS cands = new MaxHeapS(128);
    float minAcc = Float.NEGATIVE_INFINITY;

    for (int seed : seeds) {
      if (visited.get(seed)) {
        continue;
      }
      visited.set(seed);
      if (dc >= budget) {
        r.earlyTerminated = true;
        return finish(r, dc, visited, worst, expanded);
      }
      float s = scorer.score(seed);
      dc++;
      cands.add(seed, s);
      if (acceptOrds == null || acceptOrds.get(seed)) {
        if (worst.size < k || s > worst.peekScore()) {
          worst.add(seed, s);
          if (worst.size > k) {
            worst.poll();
          }
          r.lastImproveAt = dc;
          recordImprove(r, dc);
        }
        if (worst.size == k) {
          minAcc = Math.max(minAcc, worst.peekScore());
        }
      }
    }

    while (!cands.isEmpty()) {
      if (dc >= budget) {
        r.earlyTerminated = true;
        return finish(r, dc, visited, worst, expanded);
      }
      float topS = cands.peekScore();
      if (topS < minAcc) {
        break;
      }
      int node = cands.poll();
      expanded++;
      graph.seek(0, node);
      int n;
      while ((n = graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
        if (visited.get(n)) {
          continue;
        }
        visited.set(n);
        if (dc >= budget) {
          r.earlyTerminated = true;
          return finish(r, dc, visited, worst, expanded);
        }
        float s = scorer.score(n);
        dc++;
        if (s > minAcc) {
          cands.add(n, s);
          if (acceptOrds == null || acceptOrds.get(n)) {
            if (worst.size < k || s > worst.peekScore()) {
              worst.add(n, s);
              if (worst.size > k) {
                worst.poll();
              }
              r.lastImproveAt = dc;
              recordImprove(r, dc);
              if (worst.size == k) {
                minAcc = worst.peekScore();
              }
            }
          }
        }
      }
    }

    return finish(r, dc, visited, worst, expanded);
  }

  /** Descent starting from {@code startOrd} at level {@code startLevel}
   *  (restart from a non-top-level node: descend only through levels ≤ startLevel). */
  public static Result searchFrom(HnswGraph graph, int size, ScorerFn scorer, int k,
                                  int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                                  int startOrd, int startLevel) throws IOException {
    return search(graph, size, scorer, k, visitLimit, acceptOrds, startOrd, new int[0],
        Math.max(0, Math.min(startLevel, graph.numLevels() - 1)), null,
        Float.NEGATIVE_INFINITY);
  }

  /** Lucene eps[]-semantics proxy: ONE search with multiple base-layer seeds and a
   *  SHARED budget + shared collector (the production multi-seed mechanism). */
  public static Result searchMultiSeed(HnswGraph graph, int size, ScorerFn scorer, int k,
                                       int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                                       int[] extraSeeds) throws IOException {
    return search(graph, size, scorer, k, visitLimit, acceptOrds, graph.entryNode(),
        extraSeeds, graph.numLevels() - 1, null, Float.NEGATIVE_INFINITY);
  }

  /** Path-marked restart: {@code preVisited} nodes are skipped entirely (never
   *  re-scored) — the search only pays for new territory. This mirrors the
   *  shared-visited semantics of Lucene's eps[] mechanism, applied sequentially. */
  public static Result searchFrom(HnswGraph graph, int size, ScorerFn scorer, int k,
                                  int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                                  int startOrd, int startLevel, BitSet preVisited)
      throws IOException {
    return search(graph, size, scorer, k, visitLimit, acceptOrds, startOrd, new int[0],
        Math.max(0, Math.min(startLevel, graph.numLevels() - 1)), preVisited,
        Float.NEGATIVE_INFINITY);
  }

  /** Shared-threshold variant of the gated beam search: the frontier gate starts
   *  from {@code minAccSeed} (the union's current k-th score) instead of −∞, and
   *  the threshold only ever tightens (max of seed and this beam's own top-k).
   *  A round seeded this way digs until its frontier can no longer beat the
   *  UNION's k-th result — it exits early only when it is genuinely useless for
   *  the union, not when it is locally satisfied. This is the port analog of
   *  Lucene's collector-driven minCompetitiveSimilarity (TopScoreDocCollector
   *  created with a score floor), i.e. the deployable "avoid CDSM early exit". */
  public static Result searchFromSeeded(HnswGraph graph, int size, ScorerFn scorer, int k,
                                        int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                                        int startOrd, int startLevel, BitSet preVisited,
                                        float minAccSeed)
      throws IOException {
    return search(graph, size, scorer, k, visitLimit, acceptOrds, startOrd, new int[0],
        Math.max(0, Math.min(startLevel, graph.numLevels() - 1)), preVisited, minAccSeed);
  }

  /** GATE-FREE variant of the beam search (faiss/hnswlib semantics): the frontier
   *  gate (`topS < minAcc → break`) and the candidate-acceptance gate (`s > minAcc`)
   *  are both removed — every unvisited neighbor is scored and added to the
   *  candidate list until the hard dc budget is spent or the frontier empties.
   *  Identical in every other respect (visited set, marking, descent, accept
   *  filter, dc accounting). Used to isolate "the gate" from "the coverage
   *  principle" in the mechanism anatomy. */
  public static Result searchUngated(HnswGraph graph, int size, ScorerFn scorer, int k,
                                     int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                                     int startOrd, int startLevel, BitSet preVisited)
      throws IOException {
    Result r = new Result();
    long dc = 0;
    long budget = visitLimit > 0 ? visitLimit : Long.MAX_VALUE;
    BitSet visited = new BitSet(size);
    if (preVisited != null) {
      visited.or(preVisited);
    }
    MinHeapS worst = new MinHeapS(k + 1);
    int expanded = 0;

    int eps = startOrd;
    r.entryNode = startOrd;
    float bestScore = scorer.score(eps);
    dc++;
    r.bestImproveAt = dc;

    for (int level = startLevel; level >= 1 && dc < budget; level--) {
      boolean changed = true;
      visited.set(eps);
      while (changed) {
        changed = false;
        graph.seek(level, eps);
        int n;
        while ((n = graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
          if (visited.get(n)) {
            continue;
          }
          visited.set(n);
          if (dc >= budget) {
            r.earlyTerminated = true;
            return finish(r, dc, visited, worst, expanded);
          }
          float s = scorer.score(n);
          dc++;
          if (s > bestScore) {
            bestScore = s;
            eps = n;
            changed = true;
            r.bestImproveAt = dc;
          }
        }
      }
    }
    r.descendedNode = eps;

    visited.clear();
    if (preVisited != null) {
      visited.or(preVisited);
    }
    MaxHeapS cands = new MaxHeapS(128);

    float s0 = scorer.score(eps);
    dc++;
    visited.set(eps);
    cands.add(eps, s0);
    if (acceptOrds == null || acceptOrds.get(eps)) {
      worst.add(eps, s0);
      if (worst.size > k) {
        worst.poll();
      }
      r.lastImproveAt = dc;
      recordImprove(r, dc);
    }

    while (!cands.isEmpty()) {
      if (dc >= budget) {
        r.earlyTerminated = true;
        return finish(r, dc, visited, worst, expanded);
      }
      int node = cands.poll();
      expanded++;
      graph.seek(0, node);
      int n;
      while ((n = graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
        if (visited.get(n)) {
          continue;
        }
        visited.set(n);
        if (dc >= budget) {
          r.earlyTerminated = true;
          return finish(r, dc, visited, worst, expanded);
        }
        float s = scorer.score(n);
        dc++;
        cands.add(n, s);            // gate-free: every neighbor joins the frontier
        if (acceptOrds == null || acceptOrds.get(n)) {
          if (worst.size < k || s > worst.peekScore()) {
            worst.add(n, s);
            if (worst.size > k) {
              worst.poll();
            }
            r.lastImproveAt = dc;
            recordImprove(r, dc);
          }
        }
      }
    }
    return finish(r, dc, visited, worst, expanded);
  }

  /** Gate-free search from the graph's own entry node (faiss-style single). */
  public static Result searchUngated(HnswGraph graph, int size, ScorerFn scorer, int k,
                                     int visitLimit, org.apache.lucene.util.Bits acceptOrds)
      throws IOException {
    return searchUngated(graph, size, scorer, k, visitLimit, acceptOrds, graph.entryNode(),
        graph.numLevels() - 1, null);
  }

  private static Result search(HnswGraph graph, int size, ScorerFn scorer, int k,
                               int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                               int startOrd, int[] extraSeeds)
      throws IOException {
    return search(graph, size, scorer, k, visitLimit, acceptOrds, startOrd, extraSeeds,
        graph.numLevels() - 1, null, Float.NEGATIVE_INFINITY);
  }

  private static Result search(HnswGraph graph, int size, ScorerFn scorer, int k,
                               int visitLimit, org.apache.lucene.util.Bits acceptOrds,
                               int startOrd, int[] extraSeeds, int startLevel,
                               BitSet preVisited, float minAccSeed)
      throws IOException {
    Result r = new Result();
    long dc = 0;
    long budget = visitLimit > 0 ? visitLimit : Long.MAX_VALUE;
    BitSet visited = new BitSet(size);
    if (preVisited != null) {
      visited.or(preVisited);   // path marking: never re-score explored nodes
    }
    // result set keeps the k BEST; the WORST (smallest score) sits on top for
    // eviction -> MIN-heap by score
    MinHeapS worst = new MinHeapS(k + 1);
    int expanded = 0;

    // ---------- findBestEntryPoint ----------
    int eps = startOrd;
    r.entryNode = startOrd;
    float bestScore = scorer.score(eps);
    dc++;
    r.bestImproveAt = dc;
    int topLevel = graph.numLevels() - 1;

    for (int level = startLevel; level >= 1 && dc < budget; level--) {
      boolean changed = true;
      visited.set(eps);
      while (changed) {
        changed = false;
        graph.seek(level, eps);
        int n;
        while ((n = graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
          if (visited.get(n)) {
            continue;
          }
          visited.set(n);
          if (dc >= budget) {
            r.earlyTerminated = true;
            return finish(r, dc, visited, worst, expanded);
          }
          float s = scorer.score(n);
          dc++;
          if (s > bestScore) {
            bestScore = s;
            eps = n;
            changed = true;
            r.bestImproveAt = dc;
          }
        }
      }
    }
    r.descendedNode = eps;

    // ---------- searchLevel(0) ----------
    visited.clear();
    if (preVisited != null) {
      visited.or(preVisited);   // keep the marked territory across the transition
    }
    MaxHeapS cands = new MaxHeapS(128);        // best (largest score) on top
    float minAcc = minAccSeed;                 // accept only s > minAcc (seeded = union k-th)

    int numSeeds = 1 + extraSeeds.length;
    int[] seeds = new int[numSeeds];
    seeds[0] = eps;
    System.arraycopy(extraSeeds, 0, seeds, 1, extraSeeds.length);

    for (int seed : seeds) {
      if (visited.get(seed)) {
        continue;
      }
      visited.set(seed);
      if (dc >= budget) {
        r.earlyTerminated = true;
        return finish(r, dc, visited, worst, expanded);
      }
      float s = scorer.score(seed);
      dc++;
      cands.add(seed, s);
      if (acceptOrds == null || acceptOrds.get(seed)) {
        if (worst.size < k || s > worst.peekScore()) {
          worst.add(seed, s);
          if (worst.size > k) {
            worst.poll();
          }
          r.lastImproveAt = dc;
          recordImprove(r, dc);
        }
        if (worst.size == k) {
          minAcc = Math.max(minAcc, worst.peekScore());
        }
      }
    }

    while (!cands.isEmpty()) {
      if (dc >= budget) {
        r.earlyTerminated = true;
        return finish(r, dc, visited, worst, expanded);
      }
      float topS = cands.peekScore();
      if (topS < minAcc) {          // best candidate cannot beat worst result
        // gate closure: capture the frontier — the boundary of the threshold
        // subgraph (nodes scored but never expanded; their unvisited neighbors
        // are the valley entrances the gate cut off).
        r.frontier = java.util.Arrays.copyOf(cands.ords, cands.size);
        r.frontierScores = java.util.Arrays.copyOf(cands.scores, cands.size);
        break;
      }
      int node = cands.poll();
      expanded++;
      graph.seek(0, node);
      int n;
      while ((n = graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
        if (visited.get(n)) {
          continue;
        }
        visited.set(n);
        if (dc >= budget) {
          r.earlyTerminated = true;
          return finish(r, dc, visited, worst, expanded);
        }
        float s = scorer.score(n);
        dc++;
        if (s > minAcc) {           // friendScore > minAcceptedSimilarity
          cands.add(n, s);          // bridges even when not accepted
          if (acceptOrds == null || acceptOrds.get(n)) {
            if (worst.size < k || s > worst.peekScore()) {
              worst.add(n, s);
              if (worst.size > k) {
                worst.poll();
              }
              r.lastImproveAt = dc;
              recordImprove(r, dc);
              if (worst.size == k) {
                minAcc = Math.max(minAcc, worst.peekScore());   // tighten the gate (never below seed)
              }
            }
          }
        }
      }
    }

    return finish(r, dc, visited, worst, expanded);
  }

  /** Append an improvement step to the trace (small counts, simple growth). */
  private static void recordImprove(Result r, long dc) {
    int[] t = r.improveTrace;
    int[] nt = java.util.Arrays.copyOf(t, t.length + 1);
    nt[t.length] = (int) dc;
    r.improveTrace = nt;
  }

  private static Result finish(Result r, long dc, BitSet visited, MinHeapS worst,
                               int expanded) {
    r.scoreComps = dc;
    r.numVisited = visited.cardinality();
    r.numExpanded = expanded;
    r.visited = visited;
    int cnt = worst.size;
    int[] ords = new int[cnt];
    float[] ss = new float[cnt];
    // pop worst (smallest) first -> ascending -> reverse to descending score order
    for (int i = cnt - 1; i >= 0; i--) {
      ords[i] = worst.ords[0];
      ss[i] = worst.scores[0];
      worst.poll();
    }
    r.topK = ords;
    r.topKScores = ss;
    return r;
  }

  /** Max-heap by score: largest on top (best candidate pops first). */
  static final class MaxHeapS {
    int[] ords;
    float[] scores;
    int size;

    MaxHeapS(int cap) {
      ords = new int[cap];
      scores = new float[cap];
    }

    boolean isEmpty() {
      return size == 0;
    }

    float peekScore() {
      return scores[0];
    }

    void add(int ord, float s) {
      if (size == ords.length) {
        ords = java.util.Arrays.copyOf(ords, size * 2);
        scores = java.util.Arrays.copyOf(scores, size * 2);
      }
      ords[size] = ord;
      scores[size] = s;
      siftUp(size);
      size++;
    }

    int poll() {
      int top = ords[0];
      size--;
      if (size > 0) {
        ords[0] = ords[size];
        scores[0] = scores[size];
        siftDown(0);
      }
      return top;
    }

    private void siftUp(int i) {
      while (i > 0) {
        int p = (i - 1) / 2;
        if (scores[p] < scores[i]) {
          swap(p, i);
          i = p;
        } else {
          break;
        }
      }
    }

    private void siftDown(int i) {
      while (true) {
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        int m = i;
        if (l < size && scores[l] > scores[m]) m = l;
        if (r < size && scores[r] > scores[m]) m = r;
        if (m == i) break;
        swap(i, m);
        i = m;
      }
    }

    private void swap(int a, int b) {
      int to = ords[a];
      ords[a] = ords[b];
      ords[b] = to;
      float ts = scores[a];
      scores[a] = scores[b];
      scores[b] = ts;
    }
  }

  /** Min-heap by score: smallest on top (worst result for eviction). */
  static final class MinHeapS {
    int[] ords;
    float[] scores;
    int size;

    MinHeapS(int cap) {
      ords = new int[cap];
      scores = new float[cap];
    }

    float peekScore() {
      return scores[0];
    }

    void add(int ord, float s) {
      if (size == ords.length) {
        ords = java.util.Arrays.copyOf(ords, size * 2);
        scores = java.util.Arrays.copyOf(scores, size * 2);
      }
      ords[size] = ord;
      scores[size] = s;
      siftUp(size);
      size++;
    }

    int poll() {
      int top = ords[0];
      size--;
      if (size > 0) {
        ords[0] = ords[size];
        scores[0] = scores[size];
        siftDown(0);
      }
      return top;
    }

    private void siftUp(int i) {
      while (i > 0) {
        int p = (i - 1) / 2;
        if (scores[p] > scores[i]) {
          swap(p, i);
          i = p;
        } else {
          break;
        }
      }
    }

    private void siftDown(int i) {
      while (true) {
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        int m = i;
        if (l < size && scores[l] < scores[m]) m = l;
        if (r < size && scores[r] < scores[m]) m = r;
        if (m == i) break;
        swap(i, m);
        i = m;
      }
    }

    private void swap(int a, int b) {
      int to = ords[a];
      ords[a] = ords[b];
      ords[b] = to;
      float ts = scores[a];
      scores[a] = scores[b];
      scores[b] = ts;
    }
  }
}
