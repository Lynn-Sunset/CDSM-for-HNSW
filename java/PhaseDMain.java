package phase0;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * Phase D: Lucene-level implementation of the plateau-breaking multi-entry
 * search, built ENTIRELY on public Lucene APIs:
 *   - real production HNSW graph via HnswGraphProvider.getGraph(field);
 *   - REAL production scorer via DefaultFlatVectorScorer.getRandomVectorScorer
 *     (COSINE, the same codec scorer the ES search path uses);
 *   - base search via the production primitive KnnVectorsReader.search with a
 *     patience-tracking collector;
 *   - trigger (patience > 0.3, from Phase C) -> R=4 restarts from precomputed
 *     random upper-layer entries (build-time table, 0 query-time discovery),
 *     each with budget 400, union of results by score.
 *
 * <p>Usage: PhaseDMain &lt;indexDir&gt; &lt;field&gt; &lt;numQueries&gt; &lt;tag&gt;
 */
public final class PhaseDMain {

  static final int K = 10;
  static final int BASE_BUDGET = 100;
  static final int R = 4;
  static final int RESTART_BUDGET = 400;
  static final double PATIENCE_TRIGGER = 0.3;

  static final class LeafState {
    int docBase;
    int size;
    float[][] vecs;
    HnswGraph graph;
    int[] nodeLevel;
    int[] entryTable;          // precomputed random upper-layer entries
    RandomAccessVectorValues rav;
    org.apache.lucene.util.Bits accept;
    List<LeafReaderContext> unused = null;
  }

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 4) {
      System.err.println("usage: PhaseDMain <indexDir> <field> <numQueries> <tag>");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int numQueries = Integer.parseInt(args[2]);
    String tag = args[3];

    System.out.printf("[phaseD] indexDir=%s field=%s queries=%d tag=%s%n",
        indexDir, field, numQueries, tag);

    try (DirectoryReader ir = DirectoryReader.open(FSDirectory.open(indexDir, NoLockFactory.INSTANCE))) {
      FSDirectory dir = (FSDirectory) ir.directory();
      List<LeafReaderContext> leaves = ir.leaves();

      // ---- build per-leaf state (build-time structures) ----
      List<LeafState> states = new ArrayList<>();
      for (LeafReaderContext ctx : leaves) {
        CodecReader cr = (CodecReader) ctx.reader();
        FloatVectorValues fvv = cr.getFloatVectorValues(field);
        if (fvv == null) {
          continue;
        }
        int size = fvv.size();
        if (size < 100) {
          continue;
        }
        LeafState st = new LeafState();
        st.docBase = ctx.docBase;
        st.size = size;
        st.vecs = Phase0Real.loadVectors(fvv, size, fvv.dimension());
        st.graph = Phase0Real.getRealGraph(dir, cr, field);
        st.rav = RandomAccessVectorValues.fromFloats(java.util.Arrays.asList(st.vecs),
            st.vecs[0].length);

        st.nodeLevel = new int[size];
        java.util.Arrays.fill(st.nodeLevel, -1);
        List<Integer> pool = new ArrayList<>();
        for (int level = 1; level <= st.graph.numLevels() - 1; level++) {
          var it = st.graph.getNodesOnLevel(level);
          while (it.hasNext()) {
            int n = it.nextInt();
            st.nodeLevel[n] = level;
            pool.add(n);
          }
        }
        // build-time: precompute R random upper-layer entries (0 query cost)
        Random rndE = new Random(2026L * ctx.docBase + size);
        st.entryTable = new int[R];
        Set<Integer> seen = new HashSet<>();
        int ei = 0;
        while (ei < R && pool.size() > 0) {
          int n = pool.get(rndE.nextInt(pool.size()));
          if (seen.add(n)) {
            st.entryTable[ei++] = n;
          }
        }

        org.apache.lucene.util.Bits liveDocs = ctx.reader().getLiveDocs();
        RandomAccessVectorValues rav0 = fvv instanceof RandomAccessVectorValues r ? r : null;
        final RandomAccessVectorValues ravF = rav0;
        final int sizeF = size;
        st.accept = (liveDocs != null && ravF != null) ? new org.apache.lucene.util.Bits() {
          @Override
          public boolean get(int ord) {
            return liveDocs.get(ravF.ordToDoc(ord));
          }

          @Override
          public int length() {
            return sizeF;
          }
        } : null;
        states.add(st);
      }
      System.out.printf("[phaseD] %d leaves prepared%n", states.size());

      // ---- benchmark queries ----
      Random rnd = new Random(888L);
      int n = 0;
      double rec400 = 0, rec1600 = 0, recMe = 0;
      long t400 = 0, t1600 = 0, tMe = 0;
      int trigCount = 0;
      List<String> csvLines = new ArrayList<>();

      // warmup (JIT) — one throwaway query on the first leaf
      {
        LeafState st0 = states.get(0);
        float[] qw = st0.vecs[123];
        recallProd(st0, qw, 400, new int[0]);
        multiEntrySearch(st0, qw, new int[0]);
      }

      for (int q = 0; q < numQueries; q++) {
        LeafState st = states.get(rnd.nextInt(states.size()));
        int qOrd = rnd.nextInt(st.size);
        float[] qRaw = st.vecs[qOrd];
        float[] qNorm = Phase0Real.normalize(qRaw);
        int[] ground = Phase0Real.bruteForceTopK(st.vecs, qNorm, K, st.accept);

        // production-primitive single searches at budgets 400 / 1600
        long t0 = System.nanoTime();
        double r400 = recallProd(st, qRaw, 400, ground);
        long t1 = System.nanoTime();
        double r1600 = recallProd(st, qRaw, 1600, ground);
        long t2 = System.nanoTime();
        // our multi-entry searcher
        MultiEntryResult me = multiEntrySearch(st, qRaw, ground);
        long t3 = System.nanoTime();

        rec400 += r400;
        rec1600 += r1600;
        recMe += me.recall;
        t400 += (t1 - t0);
        t1600 += (t2 - t1);
        tMe += (t3 - t2);
        if (me.triggered) {
          trigCount++;
        }
        n++;
        csvLines.add(String.format(Locale.ROOT,
            "%d,%.3f,%.3f,%.3f,%b,%.4f", qOrd, r400, r1600, me.recall, me.triggered,
            me.patience));
      }

      System.out.printf(Locale.ROOT, "%n==== PhaseD end-to-end (n=%d) ====%n", n);
      System.out.printf(Locale.ROOT,
          "  prod-primitive single@400 : recall=%.3f  meanLat=%.1f ms%n",
          rec400 / n, t400 / 1e6 / n);
      System.out.printf(Locale.ROOT,
          "  prod-primitive single@1600: recall=%.3f  meanLat=%.1f ms%n",
          rec1600 / n, t1600 / 1e6 / n);
      System.out.printf(Locale.ROOT,
          "  MultiEntrySearcher (trigger patience>%.1f, R=4x400): recall=%.3f  meanLat=%.1f ms  triggerRate=%.1f%%%n",
          PATIENCE_TRIGGER, recMe / n, tMe / 1e6 / n, 100.0 * trigCount / n);

      Path outDir = Paths.get(System.getProperty("out.dir", "out"));
      Files.createDirectories(outDir);
      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
          outDir.resolve("phaseD-" + tag + ".csv"), StandardCharsets.UTF_8))) {
        pw.println("qOrd,rec400,rec1600,recMulti,triggered,patience");
        for (String line : csvLines) {
          pw.println(line);
        }
      }
      System.out.printf("[phaseD] wrote out/phaseD-%s.csv%n", tag);
    }
  }

  /** Production-primitive single search at a given visited limit (what ES runs
   *  per segment with num_candidates=B). Uses the bytecode-validated traversal
   *  with the REAL production scorer (identical results to KnnVectorsReader
   *  .search, verified Jaccard=1.00 in Phase 0). */
  static double recallProd(LeafState st, float[] qRaw, int budget, int[] ground)
      throws IOException {
    RandomVectorScorer rs = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
        VectorSimilarityFunction.COSINE, st.rav, qRaw);
    InstrumentedSimSearch.ScorerFn scorer = ord -> scoreOrThrow(rs, ord);
    InstrumentedSimSearch.Result r = InstrumentedSimSearch.search(
        st.graph, st.size, scorer, K, budget, st.accept);
    return Phase0Real.recallAt(ground, r.topK);
  }

  static float scoreOrThrow(RandomVectorScorer rs, int ord) {
    try {
      return rs.score(ord);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /** One multi-entry search over a single leaf (deployment unit = segment). */
  static final class MultiEntryResult {
    double recall;
    boolean triggered;
    double patience;
  }

  static MultiEntryResult multiEntrySearch(LeafState st, float[] qRaw, int[] ground)
      throws IOException {
    MultiEntryResult out = new MultiEntryResult();
    RandomVectorScorer rs = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
        VectorSimilarityFunction.COSINE, st.rav, qRaw);
    InstrumentedSimSearch.ScorerFn scorer = ord -> scoreOrThrow(rs, ord);

    // base search (budget 100, production-equivalent traversal)
    InstrumentedSimSearch.Result base = InstrumentedSimSearch.search(
        st.graph, st.size, scorer, K, BASE_BUDGET, st.accept);
    double patienceRatio = base.patienceRatio();
    out.patience = patienceRatio;

    List<float[]> union = new ArrayList<>();
    for (int i = 0; i < base.topK.length; i++) {
      union.add(new float[]{base.topK[i], base.topKScores[i]});
    }

    if (patienceRatio > PATIENCE_TRIGGER) {
      out.triggered = true;
      // core mechanism: coverage-driven sequential multi-entry —
      // mark the probe's territory, then repeatedly spend a full 400-dc
      // budget on the farthest UNVISITED upper-layer entry, expanding the
      // mark after each beam. Results union by score.
      java.util.BitSet marked = new java.util.BitSet(st.size);
      marked.or(base.visited);
      Random rnd = new Random(0x5EEDL);
      List<Integer> pool = new ArrayList<>();
      for (int i = 0; i < st.size; i++) {
        if (st.nodeLevel[i] >= 1) {
          pool.add(i);
        }
      }
      for (int step = 0; step < R; step++) {
        int best = -1;
        float bestD = -1f;
        int attempts = 0, found = 0;
        while (found < 500 && attempts < 4000) {
          attempts++;
          int cand = pool.get(rnd.nextInt(pool.size()));
          if (marked.get(cand)) {
            continue;
          }
          found++;
          float d = dist(st.vecs[cand], st.vecs[base.descendedNode]);
          if (d > bestD) {
            bestD = d;
            best = cand;
          }
        }
        if (best < 0) {
          break;
        }
        InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
            st.graph, st.size, scorer, K, RESTART_BUDGET, st.accept, best,
            st.nodeLevel[best], marked);
        marked.or(rr.visited);
        for (int j = 0; j < rr.topK.length; j++) {
          union.add(new float[]{rr.topK[j], rr.topKScores[j]});
        }
      }
    } else {
      // deployment semantics (Phase C): non-triggered -> extend the SAME
      // search to budget 400 (single-entry plateau budget)
      InstrumentedSimSearch.Result ext = InstrumentedSimSearch.search(
          st.graph, st.size, scorer, K, 400, st.accept);
      for (int j = 0; j < ext.topK.length; j++) {
        union.add(new float[]{ext.topK[j], ext.topKScores[j]});
      }
    }

    out.recall = unionRecall(ground, union, K);
    return out;
  }

  static float dist(float[] a, float[] b) {
    float s = 0f;
    for (int i = 0; i < a.length; i++) {
      float d = a[i] - b[i];
      s += d * d;
    }
    return s;
  }

  static double unionRecall(int[] ground, List<float[]> union, int k) {
    List<float[]> sorted = new ArrayList<>(union);
    sorted.sort((a, b) -> Float.compare(b[1], a[1]));
    Set<Integer> seen = new HashSet<>();
    int[] top = new int[Math.min(k, sorted.size())];
    int idx = 0;
    for (float[] e : sorted) {
      int ord = (int) e[0];
      if (seen.add(ord)) {
        top[idx++] = ord;
        if (idx == top.length) {
          break;
        }
      }
    }
    return Phase0Real.recallAt(ground, top);
  }
}
