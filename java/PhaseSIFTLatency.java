package phase0;

import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;

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
 * SIFT1M wall-clock control: the same fairness sweep as PhaseT2ILatency,
 * on the homogeneous classic corpus (128-d, L2, 1M nodes, in-memory graph).
 * Query sampling identical to PhaseSIFT/Table 1 (Random(777)).
 *
 * <p>Both entry forms measured (lite = build-time random table; farthest =
 * 500-sample farthest-unvisited scan) because Table 1's SIFT result used the
 * farthest-sampled form — the sweep decides whether the probe-400 mark also
 * makes entry choice second-order on uniform data.
 *
 * <p>Configs: single-{100,200,400,800,1600}; mkL-R1/R2/R4(+SG); mkF-R1/R2/R4(+SG);
 * mkF-R4x200-SG. Harness: 2 warmup passes, 7 rounds, per-query medians,
 * p50/p90/p99. One scorer per query shared across configs.
 *
 * <p>Usage: PhaseSIFTLatency &lt;dataDir&gt; &lt;numQueries&gt; &lt;tag&gt;
 */
public final class PhaseSIFTLatency {

  static final int K = 10;
  static final int M = 16;
  static final int EF_C = 100;
  static final int PROBE = 400;
  static final int ROUNDS = 7;
  static final int WARMUP_PASSES = 2;

  static final class Cfg {
    final String name;
    final int kind;          // 0 = single(budget); 1 = marked-lite; 2 = marked-farthest
    final int budget;
    final int rounds;
    final int perBudget;
    final boolean scoreGate;

    Cfg(String name, int budget) {
      this.name = name; this.kind = 0; this.budget = budget;
      this.rounds = 0; this.perBudget = 0; this.scoreGate = false;
    }

    Cfg(String name, int kind, int rounds, int perBudget, boolean scoreGate) {
      this.name = name; this.kind = kind; this.budget = 0;
      this.rounds = rounds; this.perBudget = perBudget; this.scoreGate = scoreGate;
    }
  }

  static final Cfg[] CFGS = {
      new Cfg("single-100", 100),
      new Cfg("single-200", 200),
      new Cfg("single-400", 400),
      new Cfg("single-800", 800),
      new Cfg("single-1600", 1600),
      new Cfg("mkL-R1x400", 1, 1, 400, false),
      new Cfg("mkL-R2x400", 1, 2, 400, false),
      new Cfg("mkL-R4x400", 1, 4, 400, false),
      new Cfg("mkL-R4x400-SG", 1, 4, 400, true),
      new Cfg("mkF-R1x400", 2, 1, 400, false),
      new Cfg("mkF-R2x400", 2, 2, 400, false),
      new Cfg("mkF-R4x400", 2, 4, 400, false),
      new Cfg("mkF-R4x400-SG", 2, 4, 400, true),
      new Cfg("mkF-R4x200-SG", 2, 4, 200, true),
  };

  static final class TimeResult {
    double recall;
    long dc;
    long ns;
    long entryNs;
  }

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 3) {
      System.err.println("usage: PhaseSIFTLatency <dataDir> <numQueries> <tag>");
      System.exit(2);
    }
    Path dir = Paths.get(args[0]);
    int numQueries = Integer.parseInt(args[1]);
    String tag = args[2];
    Path outDir = Paths.get(System.getProperty("out.dir", "out"));
    Files.createDirectories(outDir);

    System.out.printf("[sift-lat] dataDir=%s queries=%d tag=%s%n", dir, numQueries, tag);

    // ---- load ----
    float[][] base = PhaseSIFT.loadFvecs(dir.resolve("sift_base.fvecs"));
    float[][] queries = PhaseSIFT.loadFvecs(dir.resolve("sift_query.fvecs"));
    int[][] gt = PhaseSIFT.loadIvecs(dir.resolve("sift_groundtruth.ivecs"), 10);
    System.out.printf("[sift-lat] base=%dx%d queries=%d gt=%dx%d%n",
        base.length, base[0].length, queries.length, gt.length, gt[0].length);

    // ---- build (persisted: out/sift-hnsw-1M.hnsw; ~15-min build only once) ----
    Path graphFile = outDir.resolve("sift-hnsw-1M.hnsw");
    HnswGraph graph;
    if (Files.exists(graphFile)) {
      graph = PhaseHnswStore.load(graphFile);
      System.out.println("[sift-lat] HNSW loaded from cache");
    } else {
      long t0 = System.currentTimeMillis();
      graph = HnswGraphBuilder.create(new Phase0Main.FloatArraySupplier(base), M, EF_C, 42)
          .build(base.length);
      System.out.printf("[sift-lat] HNSW built in %d s (levels=%d)%n",
          (System.currentTimeMillis() - t0) / 1000, graph.numLevels());
      PhaseHnswStore.dump(graph, graphFile);
    }

    int n = base.length;
    int[] nodeLevel = new int[n];
    Arrays.fill(nodeLevel, -1);
    List<Integer> pool = new ArrayList<>();
    for (int level = 1; level <= graph.numLevels() - 1; level++) {
      var it = graph.getNodesOnLevel(level);
      while (it.hasNext()) {
        int node = it.nextInt();
        nodeLevel[node] = level;
        pool.add(node);
      }
    }
    System.out.printf("[sift-lat] upper-layer pool=%d%n", pool.size());

    int[] entryTable = new int[4];
    Random rndE = new Random(2026);
    Set<Integer> seen = new HashSet<>();
    int ei = 0;
    while (ei < 4) {
      int cand = pool.get(rndE.nextInt(pool.size()));
      if (seen.add(cand)) {
        entryTable[ei++] = cand;
      }
    }

    // ---- fixed query set: identical sampling to PhaseSIFT/Table 1 (Random(777)) ----
    int qn = Math.min(numQueries, queries.length);
    int[] qIdx = new int[qn];
    float[][] qRaw = new float[qn][];
    int[][] ground = new int[qn][];
    Random rndQ = new Random(777);
    for (int i = 0; i < qn; i++) {
      qIdx[i] = rndQ.nextInt(queries.length);
      qRaw[i] = queries[qIdx[i]];
      ground[i] = gt[qIdx[i]];
    }

    // ---- warmup ----
    for (int pass = 0; pass < WARMUP_PASSES; pass++) {
      for (int qi = 0; qi < qn; qi++) {
        final float[] qv = qRaw[qi];
        final float[][] baseF = base;
        InstrumentedSimSearch.ScorerFn scorer =
            ord -> -InstrumentedSearch.dist2(qv, baseF[ord]);
        for (Cfg cfg : CFGS) {
          run(cfg, graph, n, scorer, base, nodeLevel, pool, entryTable,
              ground[qi], new Random(777L * 131071L + qi));
        }
      }
      System.out.printf("[sift-lat] warmup pass %d/%d done%n", pass + 1, WARMUP_PASSES);
    }

    // ---- measurement ----
    int nCfg = CFGS.length;
    long[][][] times = new long[nCfg][qn][ROUNDS];
    double[][] rec = new double[nCfg][qn];
    long[][] dc = new long[nCfg][qn];
    long[] entryNsSum = new long[qn];

    for (int round = 0; round < ROUNDS; round++) {
      List<Integer> order = new ArrayList<>();
      for (int i = 0; i < nCfg; i++) {
        order.add(i);
      }
      Collections.shuffle(order, new Random(1000L + round));
      for (int qi = 0; qi < qn; qi++) {
        final float[] qv = qRaw[qi];
        final float[][] baseF = base;
        InstrumentedSimSearch.ScorerFn scorer =
            ord -> -InstrumentedSearch.dist2(qv, baseF[ord]);
        for (int c : order) {
          TimeResult tr = run(CFGS[c], graph, n, scorer, base, nodeLevel, pool, entryTable,
              ground[qi], new Random(777L * 131071L + qi));
          times[c][qi][round] = tr.ns;
          if (round == 0) {
            rec[c][qi] = tr.recall;
            dc[c][qi] = tr.dc;
            if (c == 12) {   // mkF-R4x400-SG
              entryNsSum[qi] = tr.entryNs;
            }
          }
        }
      }
      System.out.printf("[sift-lat] measurement round %d/%d done%n", round + 1, ROUNDS);
    }

    // ---- aggregates ----
    double[][] med = new double[nCfg][qn];
    for (int c = 0; c < nCfg; c++) {
      for (int qi = 0; qi < qn; qi++) {
        long[] ts = times[c][qi];
        Arrays.sort(ts);
        med[c][qi] = ts[ROUNDS / 2];
      }
    }

    System.out.printf(Locale.ROOT, "%n==== SIFT wall-clock sweep (n=%d, 7 rounds) ====%n", qn);
    System.out.printf(Locale.ROOT, "%-16s %8s %9s %9s %9s %9s%n",
        "config", "recall", "dc(mean)", "p50(ms)", "p90(ms)", "p99(ms)");
    double[] lats = new double[qn];
    double[] dcs = new double[qn];
    double[] recs = new double[qn];
    for (int c = 0; c < nCfg; c++) {
      for (int qi = 0; qi < qn; qi++) {
        lats[qi] = med[c][qi];
        dcs[qi] = dc[c][qi];
        recs[qi] = rec[c][qi];
      }
      Arrays.sort(lats);
      double rSum = 0, dSum = 0;
      for (int qi = 0; qi < qn; qi++) {
        rSum += recs[qi];
        dSum += dcs[qi];
      }
      System.out.printf(Locale.ROOT, "%-16s %8.3f %9.0f %9.3f %9.3f %9.3f%n",
          CFGS[c].name, rSum / qn, dSum / qn, lats[qn / 2] / 1e6,
          lats[(int) Math.ceil(0.90 * qn) - 1] / 1e6,
          lats[(int) Math.ceil(0.99 * qn) - 1] / 1e6);
    }
    long eSum = 0;
    for (int qi = 0; qi < qn; qi++) {
      eSum += entryNsSum[qi];
    }
    System.out.printf(Locale.ROOT,
        "  mkF-R4x400-SG entry-selection overhead: mean=%.3f ms/query%n",
        eSum / (double) qn / 1e6);

    // CSV
    List<String> csv = new ArrayList<>();
    StringBuilder hdr = new StringBuilder("q");
    for (Cfg cfg : CFGS) {
      hdr.append(',').append(cfg.name).append("-rec");
    }
    for (Cfg cfg : CFGS) {
      hdr.append(',').append(cfg.name).append("-ns");
    }
    for (Cfg cfg : CFGS) {
      hdr.append(',').append(cfg.name).append("-dc");
    }
    hdr.append(",entryNs");
    csv.add(hdr.toString());
    for (int qi = 0; qi < qn; qi++) {
      StringBuilder s = new StringBuilder().append(qi);
      for (int c = 0; c < nCfg; c++) {
        s.append(String.format(Locale.ROOT, ",%.3f", rec[c][qi]));
      }
      for (int c = 0; c < nCfg; c++) {
        s.append(',').append((long) med[c][qi]);
      }
      for (int c = 0; c < nCfg; c++) {
        s.append(',').append(dc[c][qi]);
      }
      s.append(',').append(entryNsSum[qi]);
      csv.add(s.toString());
    }
    try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
        outDir.resolve("phaseSIFT-latency-" + tag + ".csv"), StandardCharsets.UTF_8))) {
      for (String line : csv) {
        pw.println(line);
      }
    }
    System.out.printf("[sift-lat] wrote out/phaseSIFT-latency-%s.csv%n", tag);
  }

  // ---------------- execution (identical to PhaseT2ILatency) ----------------

  static TimeResult run(Cfg cfg, HnswGraph graph, int n,
                        InstrumentedSimSearch.ScorerFn scorer, float[][] base,
                        int[] nodeLevel, List<Integer> pool, int[] entryTable,
                        int[] ground, Random rnd) throws IOException {
    long t0 = System.nanoTime();
    if (cfg.kind == 0) {
      InstrumentedSimSearch.Result r = InstrumentedSimSearch.search(
          graph, n, scorer, K, cfg.budget, null);
      long t1 = System.nanoTime();
      TimeResult out = new TimeResult();
      out.ns = t1 - t0;
      out.dc = r.scoreComps;
      out.recall = Phase0Real.recallAt(ground, r.topK);
      return out;
    }

    InstrumentedSimSearch.Result probe = InstrumentedSimSearch.search(
        graph, n, scorer, K, PROBE, null);
    long dcSum = probe.scoreComps;
    long entryNs = 0;
    List<float[]> union = new ArrayList<>();
    for (int j = 0; j < probe.topK.length; j++) {
      union.add(new float[]{probe.topK[j], probe.topKScores[j]});
    }
    BitSet marked = new BitSet(n);
    marked.or(probe.visited);

    for (int step = 0; step < cfg.rounds; step++) {
      long te0 = System.nanoTime();
      int best;
      if (cfg.kind == 1) {
        best = -1;
        for (int i = 0; i < entryTable.length; i++) {
          int cand = entryTable[(step + i) % entryTable.length];
          if (!marked.get(cand)) {
            best = cand;
            break;
          }
        }
      } else {
        best = farthestUnvisited(base, pool, marked, probe.descendedNode, rnd);
      }
      long te1 = System.nanoTime();
      entryNs += te1 - te0;
      if (best < 0) {
        break;
      }
      InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
          graph, n, scorer, K, cfg.perBudget, null, best, nodeLevel[best], marked);
      dcSum += rr.scoreComps;
      marked.or(rr.visited);
      for (int j = 0; j < rr.topK.length; j++) {
        union.add(new float[]{rr.topK[j], rr.topKScores[j]});
      }
      if (cfg.scoreGate) {
        float roundMax = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < rr.topKScores.length; j++) {
          roundMax = Math.max(roundMax, rr.topKScores[j]);
        }
        float[] sc = new float[union.size()];
        for (int j = 0; j < union.size(); j++) {
          sc[j] = union.get(j)[1];
        }
        Arrays.sort(sc);
        if (roundMax <= sc[Math.max(0, sc.length - K)]) {
          break;
        }
      }
    }

    double recall = unionRecall(ground, union, K);
    long t1 = System.nanoTime();
    TimeResult out = new TimeResult();
    out.ns = t1 - t0;
    out.dc = dcSum;
    out.entryNs = entryNs;
    out.recall = recall;
    return out;
  }

  static int farthestUnvisited(float[][] base, List<Integer> pool,
                               BitSet marked, int from, Random rnd) {
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
      float d = InstrumentedSearch.dist2(base[cand], base[from]);
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
