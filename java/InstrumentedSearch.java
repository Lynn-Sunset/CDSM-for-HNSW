package phase0;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.hnsw.HnswGraph;

import java.io.IOException;
import java.util.BitSet;

/**
 * Faithful port of Lucene 9.12 {@code HnswGraphSearcher} traversal, verified
 * against the bytecode of lucene-core-9.12.0 (see lib/*.disasm.txt):
 *
 * <ul>
 *   <li>descent ({@code findBestEntryPoint}): greedy per-level walk sharing one
 *       visited BitSet across levels;</li>
 *   <li>base layer ({@code searchLevel}): beam search whose candidate frontier
 *       is gated by {@code minAcceptedSimilarity} (worst collected result),
 *       tightened dynamically; candidates is a best-first heap with no size cap;</li>
 *   <li>{@code visitedLimit} mirrors the collector's visited-count budget
 *       ({@code earlyTerminated}), i.e. the ES {@code num_candidates} analog.</li>
 * </ul>
 *
 * <p>Distance semantics: score = -squaredL2 (higher = closer). Counters:
 * {@code distComps} = distance computations; patience = distComps since last
 * top-k set improvement.
 */
public final class InstrumentedSearch {

  public static final class Result {
    public int[] topK = new int[0];
    public float[] topKDist = new float[0];
    public long distComps;         // total distance computations (descent + base)
    public long lastImproveAt;     // distComps value at last top-k set improvement
    public long bestImproveAt;     // distComps value at last best-distance improvement
    public int numVisited;         // distinct nodes scored in base layer
    public int numExpanded;        // candidate pops on base layer
    public int entryNode;          // descent start node
    public int descendedNode;      // node where descent ended
    public boolean earlyTerminated;

    public long patience() {
      return distComps - lastImproveAt;
    }

    public double patienceRatio() {
      return distComps == 0 ? 0.0 : (double) patience() / distComps;
    }
  }

  /** Squared L2 distance (ordering-equivalent to L2, cheaper). */
  public static float dist2(float[] a, float[] b) {
    float s = 0f;
    for (int i = 0; i < a.length; i++) {
      float d = a[i] - b[i];
      s += d * d;
    }
    return s;
  }

  /** Full search: descent from the graph entry + base-layer beam search. */
  public static Result search(float[] query, float[][] vectors, HnswGraph graph,
                              int k, int visitedLimit) throws IOException {
    return search(query, vectors, graph, k, visitedLimit, graph.entryNode());
  }

  /** Full search with descent starting from {@code startOrd} (restart oracle). */
  public static Result search(float[] query, float[][] vectors, HnswGraph graph,
                              int k, int visitedLimit, int startOrd) throws IOException {
    return run(query, vectors, graph, k, visitedLimit, startOrd, new int[0]);
  }

  /** Two-seed search: descent from {@code seedA}, then base-layer beam search
   *  seeded with both the descended node and {@code seedB} (Lucene's eps[]
   *  mechanism — the native form of "restart into a second basin"). */
  public static Result searchTwoSeed(float[] query, float[][] vectors, HnswGraph graph,
                                     int k, int visitedLimit, int seedA, int seedB)
      throws IOException {
    return run(query, vectors, graph, k, visitedLimit, seedA, new int[]{seedB});
  }

  private static Result run(float[] query, float[][] vectors, HnswGraph graph,
                            int k, int visitedLimit, int startOrd, int[] extraSeeds)
      throws IOException {
    Result r = new Result();
    long dc = 0;
    long budget = visitedLimit > 0 ? visitedLimit : Long.MAX_VALUE;
    int graphSize = graph.maxNodeId() + 1;
    BitSet visited = new BitSet(graphSize);
    MaxHeap results = new MaxHeap(k + 1);     // farthest (worst) on top
    int expanded = 0;

    // ---------- findBestEntryPoint (greedy descent, shared visited) ----------
    int eps = startOrd;
    r.entryNode = startOrd;
    float bestDist = dist2(query, vectors[eps]);
    dc++;
    r.bestImproveAt = dc;
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
            return finish(r, dc, visited, results, expanded);
          }
          float d = dist2(query, vectors[n]);
          dc++;
          if (d < bestDist) {
            bestDist = d;
            eps = n;
            changed = true;
            r.bestImproveAt = dc;
          }
        }
      }
    }
    r.descendedNode = eps;

    // ---------- searchLevel(0): multi-seed beam search ----------
    visited.clear();
    GrowMinHeap cands = new GrowMinHeap();    // nearest (best) on top
    float minAcc = Float.POSITIVE_INFINITY;   // gate: accept only d < minAcc

    // seed the entry point(s), mirroring the eps[] loop
    int numSeeds = 1 + extraSeeds.length;
    int[] seeds = new int[numSeeds];
    seeds[0] = eps;
    System.arraycopy(extraSeeds, 0, seeds, 1, extraSeeds.length);

    int visitedCount = 0;
    for (int seed : seeds) {
      if (visited.get(seed)) {
        continue;
      }
      visited.set(seed);
      if (dc >= budget) {
        r.earlyTerminated = true;
        return finish(r, dc, visited, results, expanded);
      }
      float d = dist2(query, vectors[seed]);   // Lucene re-scores eps on level 0
      dc++;
      visitedCount++;
      cands.add(seed, d);
      if (results.size < k || d < results.peekDist()) {
        results.add(seed, d);
        if (results.size > k) {
          results.poll();
        }
        r.lastImproveAt = dc;
      }
      if (results.size == k) {
        minAcc = results.peekDist();
      }
    }

    while (!cands.isEmpty()) {
      if (dc >= budget) {
        r.earlyTerminated = true;
        return finish(r, dc, visited, results, expanded);
      }
      float topD = cands.peekDist();
      if (topD >= minAcc) {   // best candidate cannot beat worst result
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
          return finish(r, dc, visited, results, expanded);
        }
        float d = dist2(query, vectors[n]);
        dc++;
        visitedCount++;
        if (d < minAcc) {                     // friendScore > minAcceptedSimilarity
          cands.add(n, d);
          if (results.size < k || d < results.peekDist()) {
            results.add(n, d);
            if (results.size > k) {
              results.poll();
            }
            r.lastImproveAt = dc;
            if (results.size == k) {
              minAcc = results.peekDist();    // tighten the gate
            }
          }
        }
      }
    }

    return finish(r, dc, visited, results, expanded);
  }

  private static Result finish(Result r, long dc, BitSet visited, MaxHeap results, int expanded) {
    r.distComps = dc;
    r.numVisited = visited.cardinality();
    r.numExpanded = expanded;
    int cnt = results.size;
    int[] ords = new int[cnt];
    float[] ds = new float[cnt];
    for (int i = cnt - 1; i >= 0; i--) {   // pop worst-first -> ascending output
      ords[i] = results.ords[0];
      ds[i] = results.dists[0];
      results.poll();
    }
    r.topK = ords;
    r.topKDist = ds;
    return r;
  }

  /** Growable min-heap keyed by distance: nearest (best) pops first. */
  static final class GrowMinHeap {
    int[] ords;
    float[] dists;
    int size;

    GrowMinHeap() {
      this(128);
    }

    GrowMinHeap(int cap) {
      ords = new int[cap];
      dists = new float[cap];
    }

    boolean isEmpty() {
      return size == 0;
    }

    float peekDist() {
      return dists[0];
    }

    void add(int ord, float d) {
      if (size == ords.length) {
        ords = java.util.Arrays.copyOf(ords, size * 2);
        dists = java.util.Arrays.copyOf(dists, size * 2);
      }
      ords[size] = ord;
      dists[size] = d;
      siftUp(size);
      size++;
    }

    int poll() {
      int top = ords[0];
      size--;
      if (size > 0) {
        ords[0] = ords[size];
        dists[0] = dists[size];
        siftDown(0);
      }
      return top;
    }

    private void siftUp(int i) {
      while (i > 0) {
        int p = (i - 1) / 2;
        if (dists[p] > dists[i]) {
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
        if (l < size && dists[l] < dists[m]) m = l;
        if (r < size && dists[r] < dists[m]) m = r;
        if (m == i) break;
        swap(i, m);
        i = m;
      }
    }

    private void swap(int a, int b) {
      int to = ords[a];
      ords[a] = ords[b];
      ords[b] = to;
      float td = dists[a];
      dists[a] = dists[b];
      dists[b] = td;
    }
  }

  /** Fixed-capacity max-heap keyed by distance: farthest (worst) on top. */
  static final class MaxHeap {
    final int[] ords;
    final float[] dists;
    int size;

    MaxHeap(int cap) {
      ords = new int[cap];
      dists = new float[cap];
    }

    boolean isEmpty() {
      return size == 0;
    }

    float peekDist() {
      return dists[0];
    }

    void add(int ord, float d) {
      ords[size] = ord;
      dists[size] = d;
      siftUp(size);
      size++;
    }

    int poll() {
      int top = ords[0];
      size--;
      if (size > 0) {
        ords[0] = ords[size];
        dists[0] = dists[size];
        siftDown(0);
      }
      return top;
    }

    private void siftUp(int i) {
      while (i > 0) {
        int p = (i - 1) / 2;
        if (dists[p] < dists[i]) {
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
        if (l < size && dists[l] > dists[m]) m = l;
        if (r < size && dists[r] > dists[m]) m = r;
        if (m == i) break;
        swap(i, m);
        i = m;
      }
    }

    private void swap(int a, int b) {
      int to = ords[a];
      ords[a] = ords[b];
      ords[b] = to;
      float td = dists[a];
      dists[a] = dists[b];
      dists[b] = td;
    }
  }
}
