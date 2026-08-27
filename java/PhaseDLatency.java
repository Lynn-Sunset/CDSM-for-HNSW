package phase0;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * Phase D latency v2: rigorous wall-clock benchmark + forensics.
 *
 * <p>v2 additions over v1:
 *   - ground-truth cache (out/gt-<field>.bin) — skips the 350 s brute force;
 *   - per-phase timing inside the marked mechanism (probe / entry selection /
 *     beams / union), so the cost of each component is disclosed separately;
 *   - marked-lite: the same mechanism with ZERO query-time entry-selection
 *     cost — a build-time random upper-layer entry table replaces the
 *     500-sample full-dimension farthest-unvisited scan;
 *   - raw score() micro-benchmark on the largest leaf: consecutive (cache-hot)
 *     vs random (cold mmap) ords, to separate memory effects from mechanism
 *     overhead;
 *   - leaf-size histogram, so per-leaf latency can be attributed to size.
 *
 * <p>Methods:
 *   M1 single@100 | M2 single@400 | M3 single@1600
 *   M4 marked-farthest (always-on)  — sampling farthest-unvisited, 500 samples
 *   M5 marked-farthest (triggered)  — patience&gt;0.3 gate
 *   M6 marked-lite    (always-on)  — build-time entry table, 0 selection cost
 *   M7 marked-lite    (triggered)
 */
public final class PhaseDLatency {

  static final int K = 10;
  static final int PROBE = 400;
  static final int BUDGET = 400;
  static final int R = 4;
  static final double PATIENCE_TRIGGER = 0.3;
  static final int ROUNDS = 7;
  static final int WARMUP_PASSES = 2;

  static final int M_SINGLE100 = 0;
  static final int M_SINGLE400 = 1;
  static final int M_SINGLE1600 = 2;
  static final int M_MARKED = 3;          // farthest-sampled, always-on
  static final int M_TRIGGERED = 4;       // farthest-sampled, triggered
  static final int M_LITE = 5;            // 0-cost entries, always-on
  static final int M_LITE_TRIG = 6;       // 0-cost entries, triggered
  static final int NUM_METHODS = 7;

  static final String[] M_NAMES = {
      "single@100", "single@400", "single@1600",
      "marked-farthest", "marked-farthest-trig",
      "marked-lite", "marked-lite-trig"
  };

  static final class LeafState {
    int docBase;
    int size;
    float[][] vecs;
    HnswGraph graph;
    int[] nodeLevel;
    List<Integer> pool;   // upper-layer (level>=1) node ords
    int[] entryTable;     // build-time: R random upper-layer entries (0 query cost)
    RandomAccessVectorValues rav;
    org.apache.lucene.util.Bits accept;
  }

  static final class Query {
    int leaf;
    int qOrd;
    float[] qRaw;
    int[] ground;
    long rndSeed;
  }

  static final class TimeResult {
    double recall;
    long dc;
    long ns;
    long probeNs;
    long entryNs;
    long beamNs;
    long unionNs;
    boolean triggered;
  }

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 4) {
      System.err.println("usage: PhaseDLatency <indexDir> <field> <numQueries> <tag>");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int numQueries = Integer.parseInt(args[2]);
    String tag = args[3];
    Path outDir = Paths.get(System.getProperty("out.dir", "out"));
    Files.createDirectories(outDir);
    Path gtCache = outDir.resolve("gt-" + field + ".bin");

    System.out.printf("[phaseD-lat2] indexDir=%s field=%s queries=%d tag=%s rounds=%d%n",
        indexDir, field, numQueries, tag, ROUNDS);

    try (DirectoryReader ir = DirectoryReader.open(FSDirectory.open(indexDir, NoLockFactory.INSTANCE))) {
      FSDirectory dir = (FSDirectory) ir.directory();
      List<LeafReaderContext> leaves = ir.leaves();

      // ---- per-leaf state (build-time structures, untimed) ----
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
        st.pool = new ArrayList<>();
        for (int level = 1; level <= st.graph.numLevels() - 1; level++) {
          var it = st.graph.getNodesOnLevel(level);
          while (it.hasNext()) {
            int n = it.nextInt();
            st.nodeLevel[n] = level;
            st.pool.add(n);
          }
        }
        // build-time entry table (0 query-time cost)
        Random rndE = new Random(2026L * ctx.docBase + size);
        st.entryTable = new int[R];
        Set<Integer> seen = new HashSet<>();
        int ei = 0;
        while (ei < R && st.pool.size() > 0) {
          int n = st.pool.get(rndE.nextInt(st.pool.size()));
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
      System.out.printf("[phaseD-lat2] %d leaves prepared%n", states.size());

      // leaf-size histogram (top 10)
      int[] sizes = new int[states.size()];
      for (int i = 0; i < states.size(); i++) {
        sizes[i] = states.get(i).size;
      }
      Integer[] boxed = new Integer[states.size()];
      for (int i = 0; i < boxed.length; i++) {
        boxed[i] = i;
      }
      Arrays.sort(boxed, (a, b) -> Integer.compare(states.get(b).size, states.get(a).size));
      StringBuilder sb = new StringBuilder("[phaseD-lat2] leaf sizes: ");
      for (int i = 0; i < Math.min(10, boxed.length); i++) {
        sb.append("leaf").append(boxed[i]).append('=').append(states.get(boxed[i]).size).append(' ');
      }
      System.out.println(sb);
      int bigLeaf = boxed[0];

      // ---- query set + ground truth (cached) ----
      List<Query> queries = new ArrayList<>();
      Random rnd = new Random(888L);
      boolean gtHit = loadGt(gtCache, queries, numQueries, states, rnd);
      if (gtHit) {
        System.out.printf("[phaseD-lat2] ground truth loaded from cache (%d queries)%n", queries.size());
      } else {
        long gtMs = System.currentTimeMillis();
        for (int q = 0; q < numQueries; q++) {
          LeafState st = states.get(rnd.nextInt(states.size()));
          int qOrd = rnd.nextInt(st.size);
          Query qu = new Query();
          qu.leaf = states.indexOf(st);
          qu.qOrd = qOrd;
          qu.qRaw = st.vecs[qOrd];
          qu.ground = Phase0Real.bruteForceTopK(st.vecs, Phase0Real.normalize(qu.qRaw), K, st.accept);
          qu.rndSeed = 0x5EEDL ^ ((long) st.docBase * 131071L + qOrd);
          queries.add(qu);
        }
        saveGt(gtCache, queries, states);
        System.out.printf("[phaseD-lat2] ground truth computed + cached (%d queries, %d ms)%n",
            queries.size(), System.currentTimeMillis() - gtMs);
      }

      // ---- raw score() micro-benchmark on the largest leaf ----
      {
        LeafState st = states.get(bigLeaf);
        InstrumentedSimSearch.ScorerFn scorer = scorerFor(st, st.vecs[0]);
        float sink = 0f;
        for (int i = 0; i < 500; i++) {
          sink += scorer.score(i);
        }
        long c0 = System.nanoTime();
        for (int i = 1000; i < 3000; i++) {
          sink += scorer.score(i);
        }
        long c1 = System.nanoTime();
        Random rr = new Random(42L);
        long r0 = System.nanoTime();
        for (int i = 0; i < 2000; i++) {
          sink += scorer.score(rr.nextInt(st.size));
        }
        long r1 = System.nanoTime();
        System.out.printf(Locale.ROOT,
            "[phaseD-lat2] raw score() on big leaf (size=%d): consecutive=%.1f ns/dc, random=%.1f ns/dc (sink=%.0f)%n",
            st.size, (c1 - c0) / 2000.0, (r1 - r0) / 2000.0, sink);
      }

      // ---- warmup ----
      for (int pass = 0; pass < WARMUP_PASSES; pass++) {
        for (Query qu : queries) {
          LeafState st = states.get(qu.leaf);
          InstrumentedSimSearch.ScorerFn scorer = scorerFor(st, qu.qRaw);
          for (int m = 0; m < NUM_METHODS; m++) {
            runMethod(m, st, scorer, qu, new Random(qu.rndSeed));
          }
        }
        System.out.printf("[phaseD-lat2] warmup pass %d/%d done%n", pass + 1, WARMUP_PASSES);
      }

      // ---- measurement ----
      long[][][] times = new long[NUM_METHODS][numQueries][ROUNDS];
      double[][] rec = new double[NUM_METHODS][numQueries];
      long[][] dc = new long[NUM_METHODS][numQueries];
      long[] probeNs = new long[numQueries];
      long[] entryNsArr = new long[numQueries];
      long[] beamNsArr = new long[numQueries];
      long[] unionNsArr = new long[numQueries];
      boolean[] trig = new boolean[numQueries];
      boolean[] trigLite = new boolean[numQueries];

      for (int round = 0; round < ROUNDS; round++) {
        List<Integer> order = new ArrayList<>(Arrays.asList(0, 1, 2, 3, 4, 5, 6));
        Collections.shuffle(order, new Random(1000L + round));
        for (int qi = 0; qi < numQueries; qi++) {
          Query qu = queries.get(qi);
          LeafState st = states.get(qu.leaf);
          InstrumentedSimSearch.ScorerFn scorer = scorerFor(st, qu.qRaw);
          for (int m : order) {
            TimeResult tr = runMethod(m, st, scorer, qu, new Random(qu.rndSeed));
            times[m][qi][round] = tr.ns;
            if (round == 0) {
              rec[m][qi] = tr.recall;
              dc[m][qi] = tr.dc;
              if (m == M_MARKED) {
                probeNs[qi] = tr.probeNs;
                entryNsArr[qi] = tr.entryNs;
                beamNsArr[qi] = tr.beamNs;
                unionNsArr[qi] = tr.unionNs;
              }
              if (m == M_TRIGGERED) {
                trig[qi] = tr.triggered;
              }
              if (m == M_LITE_TRIG) {
                trigLite[qi] = tr.triggered;
              }
            }
          }
        }
        System.out.printf("[phaseD-lat2] measurement round %d/%d done%n", round + 1, ROUNDS);
      }

      // ---- aggregates ----
      double[][] med = new double[NUM_METHODS][numQueries];
      double[] meanNs = new double[NUM_METHODS];
      double[] meanDc = new double[NUM_METHODS];
      double[] meanRec = new double[NUM_METHODS];
      int trigCount = 0, trigLiteCount = 0;

      List<String> csv = new ArrayList<>();
      csv.add("q,leaf,qOrd,rec100,rec400,rec1600,recFarthest,recFarthestTrig,recLite,recLiteTrig,"
          + "t100_ns,t400_ns,t1600_ns,tFarthest_ns,tFarthestTrig_ns,tLite_ns,tLiteTrig_ns,"
          + "dc100,dc400,dc1600,dcFarthest,dcFarthestTrig,dcLite,dcLiteTrig,"
          + "trigFarthest,trigLite,probeNs,entryNs,beamNs,unionNs");
      for (int qi = 0; qi < numQueries; qi++) {
        Query qu = queries.get(qi);
        StringBuilder s = new StringBuilder();
        s.append(qi).append(',').append(qu.leaf).append(',').append(qu.qOrd);
        for (int m = 0; m < NUM_METHODS; m++) {
          s.append(String.format(Locale.ROOT, ",%.3f", rec[m][qi]));
        }
        for (int m = 0; m < NUM_METHODS; m++) {
          long[] ts = times[m][qi];
          Arrays.sort(ts);
          med[m][qi] = ts[ROUNDS / 2];
          s.append(String.format(Locale.ROOT, ",%d", (long) med[m][qi]));
        }
        for (int m = 0; m < NUM_METHODS; m++) {
          s.append(',').append(dc[m][qi]);
        }
        boolean tr = trig[qi], trL = trigLite[qi];
        if (tr) {
          trigCount++;
        }
        if (trL) {
          trigLiteCount++;
        }
        s.append(',').append(tr).append(',').append(trL);
        s.append(',').append(probeNs[qi]).append(',').append(entryNsArr[qi])
            .append(',').append(beamNsArr[qi]).append(',').append(unionNsArr[qi]);
        csv.add(s.toString());

        for (int m = 0; m < NUM_METHODS; m++) {
          meanNs[m] += med[m][qi];
          meanDc[m] += dc[m][qi];
          meanRec[m] += rec[m][qi];
        }
      }

      System.out.printf(Locale.ROOT,
          "%n==== PhaseD wall-clock v2 (n=%d, %d rounds, median-of-rounds) ====%n",
          numQueries, ROUNDS);
      System.out.printf(Locale.ROOT,
          "%-22s %8s %9s %9s %9s %9s %9s%n",
          "method", "recall", "dc(mean)", "p50(ms)", "p90(ms)", "p99(ms)", "mean(ms)");
      double[] lats = new double[numQueries];
      for (int m = 0; m < NUM_METHODS; m++) {
        for (int qi = 0; qi < numQueries; qi++) {
          lats[qi] = med[m][qi];
        }
        Arrays.sort(lats);
        System.out.printf(Locale.ROOT,
            "%-22s %8.3f %9.0f %9.3f %9.3f %9.3f %9.3f%n",
            M_NAMES[m], meanRec[m] / numQueries, meanDc[m] / numQueries,
            pct(lats, 50) / 1e6, pct(lats, 90) / 1e6, pct(lats, 99) / 1e6,
            meanNs[m] / numQueries / 1e6);
      }

      // per-phase breakdown of the farthest-sampled marked mechanism
      double sProbe = 0, sEntry = 0, sBeam = 0, sUnion = 0;
      for (int qi = 0; qi < numQueries; qi++) {
        sProbe += probeNs[qi];
        sEntry += entryNsArr[qi];
        sBeam += beamNsArr[qi];
        sUnion += unionNsArr[qi];
      }
      System.out.printf(Locale.ROOT,
          "  marked-farthest phase breakdown (mean/query): probe=%.2f ms, entry=%.2f ms, beams=%.2f ms, union=%.2f ms%n",
          sProbe / numQueries / 1e6, sEntry / numQueries / 1e6,
          sBeam / numQueries / 1e6, sUnion / numQueries / 1e6);
      System.out.printf(Locale.ROOT,
          "  trigger rates: farthest=%.1f%%, lite=%.1f%% (patience>%.1f on probe@400)%n",
          100.0 * trigCount / numQueries, 100.0 * trigLiteCount / numQueries, PATIENCE_TRIGGER);

      // big-leaf-only comparison (the production-relevant setting)
      {
        double t16 = 0, tMk = 0, tLite = 0, tLiteTrig = 0;
        double r16 = 0, rMk = 0, rLite = 0, rLiteTrig = 0;
        int nb = 0;
        for (int qi = 0; qi < numQueries; qi++) {
          if (queries.get(qi).leaf == bigLeaf) {
            t16 += med[M_SINGLE1600][qi];
            tMk += med[M_MARKED][qi];
            tLite += med[M_LITE][qi];
            tLiteTrig += med[M_LITE_TRIG][qi];
            r16 += rec[M_SINGLE1600][qi];
            rMk += rec[M_MARKED][qi];
            rLite += rec[M_LITE][qi];
            rLiteTrig += rec[M_LITE_TRIG][qi];
            nb++;
          }
        }
        if (nb > 0) {
          System.out.printf(Locale.ROOT,
              "  big-leaf subset (n=%d, size=%d): single@1600 rec=%.3f lat=%.2f ms | marked-farthest rec=%.3f lat=%.2f ms | marked-lite rec=%.3f lat=%.2f ms | lite-trig rec=%.3f lat=%.2f ms%n",
              nb, states.get(bigLeaf).size, r16 / nb, t16 / nb / 1e6,
              rMk / nb, tMk / nb / 1e6, rLite / nb, tLite / nb / 1e6,
              rLiteTrig / nb, tLiteTrig / nb / 1e6);
        }
      }

      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
          outDir.resolve("phaseD-latency-" + tag + ".csv"), StandardCharsets.UTF_8))) {
        for (String line : csv) {
          pw.println(line);
        }
      }
      System.out.printf("[phaseD-lat2] wrote out/phaseD-latency-%s.csv%n", tag);
    }
  }

  // ---------------- GT cache ----------------

  static boolean loadGt(Path cache, List<Query> queries, int numQueries,
                        List<LeafState> states, Random rnd) throws IOException {
    if (!Files.exists(cache)) {
      return false;
    }
    try (DataInputStream in = new DataInputStream(Files.newInputStream(cache))) {
      if (in.readInt() != 0x6A7A) {
        return false;
      }
      if (in.readInt() != numQueries) {
        return false;
      }
      int nLeaves = in.readInt();
      if (nLeaves != states.size()) {
        return false;
      }
      for (int i = 0; i < nLeaves; i++) {
        if (in.readInt() != states.get(i).size) {
          return false;
        }
      }
      for (int q = 0; q < numQueries; q++) {
        Query qu = new Query();
        qu.leaf = in.readInt();
        qu.qOrd = in.readInt();
        LeafState st = states.get(qu.leaf);
        qu.qRaw = st.vecs[qu.qOrd];
        qu.ground = new int[K];
        for (int i = 0; i < K; i++) {
          qu.ground[i] = in.readInt();
        }
        qu.rndSeed = 0x5EEDL ^ ((long) st.docBase * 131071L + qu.qOrd);
        queries.add(qu);
        // advance the shared rnd identically to the compute path
        rnd.nextInt(states.size());
        rnd.nextInt(st.size);
      }
      return true;
    }
  }

  static void saveGt(Path cache, List<Query> queries, List<LeafState> states) throws IOException {
    try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(cache))) {
      out.writeInt(0x6A7A);
      out.writeInt(queries.size());
      out.writeInt(states.size());
      for (LeafState st : states) {
        out.writeInt(st.size);
      }
      for (Query qu : queries) {
        out.writeInt(qu.leaf);
        out.writeInt(qu.qOrd);
        for (int i = 0; i < K; i++) {
          out.writeInt(qu.ground[i]);
        }
      }
    }
  }

  // ---------------- helpers ----------------

  static double pct(double[] sorted, double p) {
    if (sorted.length == 0) {
      return 0;
    }
    int idx = (int) Math.ceil(p / 100.0 * sorted.length) - 1;
    return sorted[Math.max(0, Math.min(sorted.length - 1, idx))];
  }

  static InstrumentedSimSearch.ScorerFn scorerFor(LeafState st, float[] qRaw) throws IOException {
    RandomVectorScorer rs = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
        VectorSimilarityFunction.COSINE, st.rav, qRaw);
    return ord -> scoreOrThrow(rs, ord);
  }

  static float scoreOrThrow(RandomVectorScorer rs, int ord) {
    try {
      return rs.score(ord);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  static TimeResult runMethod(int m, LeafState st, InstrumentedSimSearch.ScorerFn scorer,
                              Query qu, Random rnd) throws IOException {
    switch (m) {
      case M_SINGLE100:
        return single(st, scorer, 100, qu.ground);
      case M_SINGLE400:
        return single(st, scorer, 400, qu.ground);
      case M_SINGLE1600:
        return single(st, scorer, 1600, qu.ground);
      case M_MARKED:
        return marked(st, scorer, qu.ground, rnd, false, false);
      case M_TRIGGERED:
        return marked(st, scorer, qu.ground, rnd, true, false);
      case M_LITE:
        return marked(st, scorer, qu.ground, rnd, false, true);
      case M_LITE_TRIG:
        return marked(st, scorer, qu.ground, rnd, true, true);
      default:
        throw new IllegalArgumentException("method " + m);
    }
  }

  static TimeResult single(LeafState st, InstrumentedSimSearch.ScorerFn scorer,
                           int budget, int[] ground) throws IOException {
    long t0 = System.nanoTime();
    InstrumentedSimSearch.Result r = InstrumentedSimSearch.search(
        st.graph, st.size, scorer, K, budget, st.accept);
    long t1 = System.nanoTime();
    TimeResult out = new TimeResult();
    out.ns = t1 - t0;
    out.dc = r.scoreComps;
    out.recall = Phase0Real.recallAt(ground, r.topK);
    return out;
  }

  static TimeResult marked(LeafState st, InstrumentedSimSearch.ScorerFn scorer,
                           int[] ground, Random rnd, boolean triggeredOnly,
                           boolean liteEntries) throws IOException {
    long tStart = System.nanoTime();
    long tProbe0 = System.nanoTime();
    InstrumentedSimSearch.Result probe = InstrumentedSimSearch.search(
        st.graph, st.size, scorer, K, PROBE, st.accept);
    long tProbe1 = System.nanoTime();
    long dc = probe.scoreComps;
    long entryNs = 0, beamNs = 0;
    boolean triggered = false;

    List<float[]> union = new ArrayList<>();
    for (int j = 0; j < probe.topK.length; j++) {
      union.add(new float[]{probe.topK[j], probe.topKScores[j]});
    }

    boolean doRounds = !triggeredOnly || probe.patienceRatio() > PATIENCE_TRIGGER;
    if (triggeredOnly && probe.patienceRatio() > PATIENCE_TRIGGER) {
      triggered = true;
    }
    if (doRounds) {
      BitSet marked = new BitSet(st.size);
      marked.or(probe.visited);
      for (int step = 0; step < R; step++) {
        long te0 = System.nanoTime();
        int best = liteEntries
            ? liteEntry(st, marked, step)
            : farthestUnvisited(st, marked, probe.descendedNode, rnd);
        long te1 = System.nanoTime();
        entryNs += te1 - te0;
        if (best < 0) {
          break;
        }
        long tb0 = System.nanoTime();
        InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
            st.graph, st.size, scorer, K, BUDGET, st.accept, best, st.nodeLevel[best], marked);
        long tb1 = System.nanoTime();
        beamNs += tb1 - tb0;
        dc += rr.scoreComps;
        marked.or(rr.visited);
        for (int j = 0; j < rr.topK.length; j++) {
          union.add(new float[]{rr.topK[j], rr.topKScores[j]});
        }
      }
    }

    long tUnion0 = System.nanoTime();
    double recall = unionRecall(ground, union, K);
    long tUnion1 = System.nanoTime();
    TimeResult out = new TimeResult();
    out.ns = tUnion1 - tStart;
    out.probeNs = tProbe1 - tProbe0;
    out.entryNs = entryNs;
    out.beamNs = beamNs;
    out.unionNs = tUnion1 - tUnion0;
    out.dc = dc;
    out.recall = recall;
    out.triggered = triggered;
    return out;
  }

  /** 0-cost entry selection: first unmarked entry from the build-time table. */
  static int liteEntry(LeafState st, BitSet marked, int step) {
    int[] table = st.entryTable;
    for (int i = 0; i < table.length; i++) {
      int cand = table[(step + i) % table.length];
      if (cand >= 0 && !marked.get(cand)) {
        return cand;
      }
    }
    return -1;
  }

  /** Farthest UNVISITED upper-layer node from the probe's descended node
   *  (500-sample scan, matches PhaseAFair.evalMarked). */
  static int farthestUnvisited(LeafState st, BitSet marked, int from, Random rnd) {
    int best = -1;
    float bestD = -1f;
    int attempts = 0, found = 0;
    List<Integer> pool = st.pool;
    while (found < 500 && attempts < 4000) {
      attempts++;
      int cand = pool.get(rnd.nextInt(pool.size()));
      if (marked.get(cand)) {
        continue;
      }
      found++;
      float d = InstrumentedSearch.dist2(st.vecs[cand], st.vecs[from]);
      if (d > bestD) {
        bestD = d;
        best = cand;
      }
    }
    return best;
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
