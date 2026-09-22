package phase0;

import java.io.*;
import java.util.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.util.hnsw.*;

/** Primitive, reusable implementation of the frozen DecisionExperiments policies.
 * A Workspace and its Leaf/graph must only be used by one thread at a time.
 * No vector or adjacency information is retained across queries.
 */
public final class CrossMetricSearch {
  public static final double GATE = .733723402;
  public record Policy(String name, String family, double parameter, int cap) {}
  public record Outcome(int[] hits, int dc, int logical, int edges, int expansions,
      int scoutDc, int scouts, long sequence, int[] trace) {}
  public record Probe(int dc, float score1, float score10, float scoreGap,
      int active, int deferred, int expansions, int edges, int upperDc,
      int baseSeen, int partialRemaining) {}
  static final int BASE = 1, COMPLETED = 2, ENTRY = 4, OPEN = 8;
  static final long INITIAL_SEQUENCE = 0xcbf29ce484222325L;

  static final class Collector extends TopKnnCollector {
    Collector(int width) { super(width, Integer.MAX_VALUE); }
    void reset() { queue.clear(); visitedCount = 0; }
  }
  static final class Frontier {
    final NeighborQueue active = new NeighborQueue(64, true);
    final NeighborQueue deferred = new NeighborQueue(64, true);
    int current = -1;
    void reset() { active.clear(); deferred.clear(); current = -1; }
  }
  public static final class Workspace {
    final PhaseT2IEndToEnd.Leaf lf;
    final VectorSimilarityFunction similarity;
    final float[] scores;
    final int[] stamp, upperStamp, cursorPos, cursorEnd;
    final byte[] flags;
    int epoch, upperEpoch;
    int[] touched = new int[1024], arena = new int[4096], diagnosticTrace = new int[1024];
    int touchedCount, arenaSize, traceSize;
    final Frontier main = new Frontier(), local = new Frontier();
    final Map<Integer, Collector> collectors = new HashMap<>();
    final Collector output = new Collector(10);
    Collector gate;
    RandomVectorScorer scorer;
    int width, dc, logical, upperDc, edgeReads, expansions, scoutDc, scoutCount;
    long sequence;
    boolean diagnostic;
    final int[] pool = new int[4];
    public final long allocationNanos;

    public Workspace(PhaseT2IEndToEnd.Leaf leaf, VectorSimilarityFunction metric) {
      long start = System.nanoTime();
      lf = leaf; similarity = metric;
      if (metric != VectorSimilarityFunction.COSINE && metric != VectorSimilarityFunction.EUCLIDEAN) throw new IllegalArgumentException("Unsupported distance metric");
      scores = new float[lf.size]; stamp = new int[lf.size]; upperStamp = new int[lf.size];
      cursorPos = new int[lf.size]; cursorEnd = new int[lf.size];
      flags = new byte[lf.size];
      allocationNanos = System.nanoTime() - start;
    }
    /** Array payload only; excludes object headers, Lucene heaps, Leaf, vectors and graph. */
    public long fixedArrayBytes() { return 21L * lf.size; }
    public long dynamicArrayBytes() { return 4L * (touched.length + arena.length + diagnosticTrace.length); }
    public int arenaCapacityInts() { return arena.length; }
    public int touchedCapacityInts() { return touched.length; }
    public void setDiagnostic(boolean enabled) { diagnostic = enabled; }
    boolean scored(int n) { return stamp[n] == epoch; }
    boolean marked(int n, int flag) { return scored(n) && (flags[n] & flag) != 0; }
    void mark(int n, int flag) { flags[n] |= flag; }

    void reset(RandomVectorScorer nextScorer, int nextWidth) {
      if (++epoch == 0) { Arrays.fill(stamp, 0); epoch = 1; }
      scorer = nextScorer; touchedCount = arenaSize = traceSize = 0;
      dc = logical = upperDc = edgeReads = expansions = scoutDc = scoutCount = 0;
      sequence = INITIAL_SEQUENCE; main.reset(); local.reset(); output.reset();
      freshGate(nextWidth);
    }
    void freshGate(int nextWidth) {
      width = nextWidth;
      gate = collectors.computeIfAbsent(nextWidth, Collector::new);
      gate.reset();
    }
    void rebuildGate(int nextWidth) {
      freshGate(Math.max(10, nextWidth));
      // Gate thresholds depend on scores, not insertion order. Lucene resolves
      // score ties by ordinal. Full trace regression verifies this replacement.
      for (int i = 0; i < touchedCount; i++) {
        int n = touched[i]; if ((flags[n] & BASE) != 0) gate.collect(n, scores[n]);
      }
    }
    void recordPhysical(int n) {
      dc++; sequence = (sequence ^ n) * 0x100000001b3L;
      if (diagnostic) {
        if (traceSize == diagnosticTrace.length) diagnosticTrace = Arrays.copyOf(diagnosticTrace, traceSize * 2);
        diagnosticTrace[traceSize++] = n;
      }
    }
    float score(int n) throws IOException {
      logical++;
      if (scored(n)) return scores[n];
      float value = scorer.score(n); recordPhysical(n);
      stamp[n] = epoch; flags[n] = 0; scores[n] = value;
      if (touchedCount == touched.length) touched = Arrays.copyOf(touched, touchedCount * 2);
      touched[touchedCount++] = n; output.collect(n, value);
      return value;
    }
    int descend(int entry, int level, int cap) throws IOException {
      int before = dc;
      if (dc >= cap && !scored(entry)) return -1;
      int ep = entry; float best = score(ep);
      if (++upperEpoch == 0) { Arrays.fill(upperStamp, 0); upperEpoch = 1; }
      for (int l = level; l >= 1 && dc < cap; l--) {
        boolean changed = true; upperStamp[ep] = upperEpoch;
        while (changed && dc < cap) {
          changed = false; lf.graph.seek(l, ep); int n;
          while ((n = lf.graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
            edgeReads++;
            if (marked(n, BASE) || upperStamp[n] == upperEpoch) continue;
            if (dc >= cap) { upperDc += dc - before; return ep; }
            upperStamp[n] = upperEpoch; float value = score(n);
            if (value > best) { best = value; ep = n; changed = true; }
          }
        }
      }
      upperDc += dc - before; return ep;
    }
    void seed(Frontier f, int ep) throws IOException {
      if (ep < 0 || marked(ep, BASE)) return;
      float value = score(ep); mark(ep, BASE); gate.collect(ep, value); f.active.add(ep, value);
    }
    void initialize(int cap) throws IOException {
      int ep = descend(lf.graph.entryNode(), lf.graph.numLevels() - 1, cap);
      if (dc < cap) seed(main, ep);
    }
    void open(int node) throws IOException {
      if (marked(node, OPEN)) return;
      lf.graph.seek(0, node); cursorPos[node] = arenaSize; int n;
      while ((n = lf.graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
        edgeReads++;
        if (arenaSize == arena.length) arena = Arrays.copyOf(arena, arenaSize * 2);
        arena[arenaSize++] = n;
      }
      cursorEnd[node] = arenaSize; mark(node, OPEN);
    }
    int advance(Frontier f, int cap, float slack) throws IOException {
      while (dc < cap) {
        if (f.current >= 0) {
          int current = f.current;
          if (cursorPos[current] == cursorEnd[current]) {
            if (marked(current, COMPLETED)) throw new AssertionError("Duplicate completion");
            mark(current, COMPLETED); expansions++; f.current = -1; continue;
          }
          int n = arena[cursorPos[current]];
          if (marked(n, BASE)) { cursorPos[current]++; continue; }
          float tau = gate.minCompetitiveSimilarity() - slack, value = score(n);
          cursorPos[current]++; mark(n, BASE);
          if (value > tau) { f.active.add(n, value); gate.collect(n, value); }
          else f.deferred.add(n, value);
          continue;
        }
        float tau = gate.minCompetitiveSimilarity() - slack;
        while (f.deferred.size() > 0 && f.deferred.topScore() > tau) {
          int n = f.deferred.pop(); if (!marked(n, COMPLETED)) f.active.add(n, scores[n]);
        }
        while (f.active.size() > 0 && marked(f.active.topNode(), COMPLETED)) f.active.pop();
        if (f.active.size() == 0) return f.deferred.size() > 0 ? 1 : 2;
        if (f.active.topScore() < tau) return 1;
        f.current = f.active.pop(); open(f.current);
      }
      return 0;
    }
    double distanceSquared(float score) {
      return similarity == VectorSimilarityFunction.EUCLIDEAN ? Math.max(0, 1.0 / score - 1.0) : Math.max(0, 1.0 - score);
    }
    int distanceAdvance(int cap, double gamma) throws IOException {
      Frontier f = main;
      while (dc < cap) {
        if (f.current >= 0) {
          int current = f.current;
          if (cursorPos[current] == cursorEnd[current]) {
            if (marked(current, COMPLETED)) throw new AssertionError("Duplicate completion");
            mark(current, COMPLETED); expansions++; f.current = -1; continue;
          }
          int n = arena[cursorPos[current]++]; if (marked(n, BASE)) continue;
          float value = score(n); mark(n, BASE); f.active.add(n, value); gate.collect(n, value); continue;
        }
        while (f.deferred.size() > 0) {
          int n = f.deferred.pop(); if (!marked(n, COMPLETED)) f.active.add(n, scores[n]);
        }
        while (f.active.size() > 0 && marked(f.active.topNode(), COMPLETED)) f.active.pop();
        if (f.active.size() == 0) return 2;
        float kth = output.minCompetitiveSimilarity();
        if (kth != Float.NEGATIVE_INFINITY && distanceSquared(f.active.topScore()) >=
            (1 + gamma) * (1 + gamma) * distanceSquared(kth)) return 1;
        f.current = f.active.pop(); open(f.current);
      }
      return 0;
    }
    void scout(int[] table, int cap, int quota, boolean base) throws IOException {
      int entry = -1;
      for (int n : table) if (!marked(n, ENTRY) && !marked(n, BASE)) { entry = n; break; }
      if (entry < 0 || dc >= cap || quota <= 0) return;
      // Old entriesUsed can mark an as-yet unscored ordinal. Preserve the mark
      // after descend's first score initializes its per-query flags.
      scoutCount++; int before = dc, stop = Math.min(cap, before + quota);
      local.reset(); int ep = descend(entry, base ? 0 : lf.nodeLevel[entry], stop);
      mark(entry, ENTRY); freshGate(10);
      if (dc < stop) seed(local, ep); advance(local, stop, 0);
      for (int n : local.active.nodes()) main.active.add(n, scores[n]);
      for (int n : local.deferred.nodes()) main.deferred.add(n, scores[n]);
      if (local.current >= 0) main.active.add(local.current, scores[local.current]);
      scoutDc += dc - before; rebuildGate(1600);
    }
    int[] candidatePool(int[] table) {
      int count = 0;
      for (int n : table) if (!marked(n, BASE) && count < 4) pool[count++] = n;
      if (count != 4) throw new AssertionError("upper candidate pool"); return pool;
    }
    Outcome finish() {
      ScoreDoc[] docs = output.topDocs().scoreDocs; int[] hits = new int[docs.length];
      for (int i = 0; i < docs.length; i++) hits[i] = docs[i].doc;
      if (dc != touchedCount) throw new AssertionError("unique scorer accounting");
      return new Outcome(hits, dc, logical, edgeReads, expansions, scoutDc, scoutCount, sequence,
          diagnostic ? Arrays.copyOf(diagnosticTrace, traceSize) : new int[0]);
    }
  }

  /** Runs the frozen common 400-distance probe. Snapshot collection scores no
   * new nodes and leaves its frontier/cursors intact in the Workspace. */
  public static Probe probe(Workspace w, float[] query) throws Exception {
    w.reset(AblationSearch.scorer(w.lf, query, w.similarity), 10);
    w.initialize(400); w.advance(w.main, 400, 0);
    float best = Float.NEGATIVE_INFINITY; int base = 0;
    for (int i = 0; i < w.touchedCount; i++) {
      int n = w.touched[i]; best = Math.max(best, w.scores[n]);
      if ((w.flags[n] & BASE) != 0) base++;
    }
    float tenth = w.output.minCompetitiveSimilarity();
    int remaining = w.main.current < 0 ? 0 : w.cursorEnd[w.main.current] - w.cursorPos[w.main.current];
    return new Probe(w.dc, best, tenth, best - tenth, w.main.active.size(),
        w.main.deferred.size(), w.expansions, w.edgeReads, w.upperDc, base, remaining);
  }

  public static Outcome policy(Workspace w, float[] query, Policy p, int[] table) throws Exception {
    return policy(w, query, p, table, 0);
  }
  public static Outcome policy(Workspace w, float[] query, Policy p, int[] table, int qid) throws Exception {
    if (p.cap <= 0) throw new IllegalArgumentException("positive cap required");
    RandomVectorScorer scorer = AblationSearch.scorer(w.lf, query, w.similarity);
    if (p.family.equals("native") || p.family.equals("nativeU")) {
      // The timed native path uses the exact historical counter, with no
      // workspace reset, hash update or per-distance diagnostic branch.
      var counter = new DecisionExperiments.Counter(scorer);
      RandomVectorScorer nativeScorer = counter;
      if (w.diagnostic) {
        w.reset(scorer, 10);
        nativeScorer = new RandomVectorScorer() {
          public float score(int n) throws IOException { float value = counter.score(n); w.recordPhysical(n); return value; }
          public int maxOrd() { return counter.maxOrd(); }
        };
      }
      int width = p.family.equals("nativeU") ? 10 : (int)p.parameter;
      var collector = new TopKnnCollector(width, p.cap) {
        @Override public float minCompetitiveSimilarity() {
          return p.family.equals("nativeU") ? Float.NEGATIVE_INFINITY : super.minCompetitiveSimilarity();
        }
      };
      HnswGraphSearcher.search(nativeScorer, collector, w.lf.graph, null);
      if (counter.calls != collector.visitedCount() || counter.calls > p.cap) throw new AssertionError("native physical calls");
      var docs = collector.topDocs().scoreDocs; int[] hits = new int[Math.min(10, docs.length)];
      for (int i = 0; i < hits.length; i++) hits[i] = docs[i].doc;
      return new Outcome(hits, counter.calls, counter.calls, -1, -1, 0, 0, 0,
          w.diagnostic ? Arrays.copyOf(w.diagnosticTrace, w.traceSize) : new int[0]);
    }
    if (!Set.of("beam", "dabs", "ungated", "main", "one", "rule", "four", "random").contains(p.family))
      throw new IllegalArgumentException("Unsupported family " + p.family);
    w.reset(scorer, p.family.equals("beam") ? (int)p.parameter : 10);
    if (Set.of("one", "rule", "four", "main", "random").contains(p.family)) {
      w.initialize(Math.min(400, p.cap)); w.advance(w.main, Math.min(400, p.cap), 0);
      double tenth = p.family.equals("rule") ? w.output.minCompetitiveSimilarity() : 0;
      if (p.family.equals("rule") && w.output.numCollected() < 10)
        throw new ArrayIndexOutOfBoundsException("Old rule requires 10 scored vectors");
      w.rebuildGate(1600);
      if (p.family.equals("one") || p.family.equals("rule")) {
        int[] pool = w.candidatePool(table); int before = w.dc;
        for (int n : pool) if (w.dc < p.cap) w.score(n);
        int screen = w.dc - before;
        if (p.family.equals("one") || tenth <= GATE)
          w.scout(new int[]{pool[0]}, p.cap, Math.max(0, p.cap / 16 - screen), false);
      } else if (p.family.equals("four")) {
        for (int i = 0; i < 4; i++) w.scout(table, p.cap, p.cap / 16, false);
      } else if (p.family.equals("random")) {
        Random rng = new Random(20260909L + qid); int node;
        do { node = rng.nextInt(w.lf.size); } while (w.marked(node, BASE));
        w.scout(new int[]{node}, p.cap, p.cap / 16, true);
      }
      w.advance(w.main, p.cap, 0);
    } else {
      w.initialize(p.cap);
      if (p.family.equals("dabs")) w.distanceAdvance(p.cap, p.parameter);
      else w.advance(w.main, p.cap, p.family.equals("beam") ? 0 : Float.POSITIVE_INFINITY);
    }
    if (w.dc > p.cap) throw new AssertionError("actual scorer cap " + p.name);
    return w.finish();
  }
}
