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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Segmented-SIFT control: the production-shape variant of the wall-clock
 * sweep. SIFT1M is split into SHARDS contiguous shards (~32k docs each, like
 * Elasticsearch segments); one HNSW per shard (M=16 efC=100); every query
 * searches EVERY shard with the per-shard budget and the shard results are
 * merged by score — the true ES per-query semantics (sum of per-segment
 * searches), unlike the single-segment sampling of the medical sweep.
 *
 * <p>Answers: (i) does segmentation turn SIFT's catastrophic queries from
 * structural (gate-welded) into starvation-type, like the re-segmented
 * medical index? (ii) does marked-lite stay the strongest tail rescuer in
 * the segmented production shape? (iii) what is the true full-query CPU
 * (dc-sum) and latency (sum vs parallel-max) cost of each tier?
 *
 * <p>Configs: single-{100,200,400,800,1600}; mkL-R1x400, mkL-R2x400,
 * mkL-R4x400, mkL-R4x400-SG, mkF-R4x400. 500 queries, Random(777) sampling,
 * 2 warmup passes, 7 rounds, per-query medians, p50/p90/p99.
 *
 * <p>Usage: PhaseSIFTSegLatency &lt;dataDir&gt; &lt;numQueries&gt; &lt;tag&gt;
 */
public final class PhaseSIFTSegLatency {

  static final int K = 10;
  static final int M = 16;
  static final int EF_C = 100;
  static final int SHARDS = 32;
  static final int PROBE = 400;
  static final int ROUNDS = 7;
  static final int WARMUP_PASSES = 2;

  static final class Cfg {
    final String name;
    final int kind;          // 0 = single(budget); 1 = mkL; 2 = mkF
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
      new Cfg("CDSM-Lite-R1x100", 1, 1, 100, false),
      new Cfg("CDSM-Lite-R2x100", 1, 2, 100, false),
      new Cfg("CDSM-Lite-R4x100", 1, 4, 100, false),
      new Cfg("CDSM-Lite-R4x100-SG", 1, 4, 100, true),
      new Cfg("CDSM-Lite-R1x200", 1, 1, 200, false),
      new Cfg("CDSM-Lite-R2x200", 1, 2, 200, false),
      new Cfg("CDSM-Lite-R4x200", 1, 4, 200, false),
      new Cfg("CDSM-Lite-R4x200-SG", 1, 4, 200, true),
      new Cfg("CDSM-Lite-R1x400", 1, 1, 400, false),
      new Cfg("CDSM-Lite-R2x400", 1, 2, 400, false),
      new Cfg("CDSM-Lite-R4x400", 1, 4, 400, false),
      new Cfg("CDSM-Lite-R4x400-SG", 1, 4, 400, true),
      new Cfg("CDSM-Far-R4x400", 2, 4, 400, false),
  };

  static final class Shard {
    int offset;               // global base ord of this shard
    float[][] vecs;
    HnswGraph graph;
    int[] nodeLevel;
    List<Integer> pool;
    int[] entryTable;
  }

  static final class ShardResult {
    List<float[]> pairs = new ArrayList<>();   // {globalOrd, score}
    long dc;
    long ns;
  }

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 3) {
      System.err.println("usage: PhaseSIFTSegLatency <dataDir> <numQueries> <tag>");
      System.exit(2);
    }
    Path dir = Paths.get(args[0]);
    int numQueries = Integer.parseInt(args[1]);
    String tag = args[2];
    Path outDir = Paths.get(System.getProperty("out.dir", "out"));
    Files.createDirectories(outDir);
    int shardCount = Integer.getInteger("shards", 32);

    System.out.printf("[sift-seg] dataDir=%s queries=%d tag=%s shards=%d%n",
        dir, numQueries, tag, shardCount);

    // ---- load ----
    float[][] base = PhaseSIFT.loadFvecs(dir.resolve("sift_base.fvecs"));
    float[][] queries = PhaseSIFT.loadFvecs(dir.resolve("sift_query.fvecs"));
    int[][] gt = PhaseSIFT.loadIvecs(dir.resolve("sift_groundtruth.ivecs"), 10);
    int n = base.length;
    int shardSize = (n + shardCount - 1) / shardCount;
    System.out.printf("[sift-seg] base=%dx%d shardSize=%d%n", n, base[0].length, shardSize);

    // ---- split + build shards (parallel) ----
    final List<Shard> shards = new ArrayList<>();
    for (int s = 0; s < shardCount; s++) {
      int from = s * shardSize;
      int to = Math.min(n, from + shardSize);
      if (from >= n) {
        break;
      }
      Shard sh = new Shard();
      sh.offset = from;
      sh.vecs = new float[to - from][];
      System.arraycopy(base, from, sh.vecs, 0, to - from);
      shards.add(sh);
    }
    System.out.printf("[sift-seg] %d shards prepared%n", shards.size());

    long t0 = System.currentTimeMillis();
    ExecutorService pool = Executors.newFixedThreadPool(8);
    List<Future<?>> futures = new ArrayList<>();
    for (int si = 0; si < shards.size(); si++) {
      final int idx = si;
      futures.add(pool.submit(() -> {
        try {
          Shard sh = shards.get(idx);
          Path shardFile = outDir.resolve("siftseg-S" + shardCount + "-shard-" + idx + ".hnsw");
          if (Files.exists(shardFile)) {
            sh.graph = PhaseHnswStore.load(shardFile);
          } else {
            sh.graph = HnswGraphBuilder.create(new Phase0Main.FloatArraySupplier(sh.vecs), M, EF_C, 42 + idx)
                .build(sh.vecs.length);
            PhaseHnswStore.dump(sh.graph, shardFile);
          }
          int sz = sh.vecs.length;
          sh.nodeLevel = new int[sz];
          Arrays.fill(sh.nodeLevel, -1);
          sh.pool = new ArrayList<>();
          for (int level = 1; level <= sh.graph.numLevels() - 1; level++) {
            var it = sh.graph.getNodesOnLevel(level);
            while (it.hasNext()) {
              int node = it.nextInt();
              sh.nodeLevel[node] = level;
              sh.pool.add(node);
            }
          }
          Random rndE = new Random(2026L * (idx + 1));
          sh.entryTable = new int[4];
          Set<Integer> seen = new HashSet<>();
          int ei = 0;
          while (ei < 4 && sh.pool.size() > 0) {
            int cand = sh.pool.get(rndE.nextInt(sh.pool.size()));
            if (seen.add(cand)) {
              sh.entryTable[ei++] = cand;
            }
          }
          System.out.printf("[sift-seg] shard %d built (%d docs, levels=%d)%n",
              idx, sz, sh.graph.numLevels());
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }));
    }
    for (Future<?> f : futures) {
      try {
        f.get();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    }
    pool.shutdown();
    try {
      pool.awaitTermination(1, TimeUnit.MINUTES);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    System.out.printf("[sift-seg] all shards built in %d s%n",
        (System.currentTimeMillis() - t0) / 1000);

    // ---- query set (Random(777), same as previous SIFT sweeps) ----
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

    int nCfg = CFGS.length;
    int nSh = shards.size();

    // ---- warmup ----
    for (int pass = 0; pass < WARMUP_PASSES; pass++) {
      for (int qi = 0; qi < qn; qi++) {
        final float[] qv = qRaw[qi];
        for (Cfg cfg : CFGS) {
          runQuery(cfg, shards, nSh, qv, ground[qi],
              new Random(777L * 131071L + qi));
        }
      }
      System.out.printf("[sift-seg] warmup pass %d/%d done%n", pass + 1, WARMUP_PASSES);
    }

    // ---- measurement ----
    long[][][] times = new long[nCfg][qn][ROUNDS];       // CPU (sum of shard ns)
    long[][][] maxTimes = new long[nCfg][qn][ROUNDS];    // parallel wall (max shard ns)
    double[][] rec = new double[nCfg][qn];
    long[][] dcSum = new long[nCfg][qn];

    for (int round = 0; round < ROUNDS; round++) {
      List<Integer> order = new ArrayList<>();
      for (int i = 0; i < nCfg; i++) {
        order.add(i);
      }
      Collections.shuffle(order, new Random(1000L + round));
      for (int qi = 0; qi < qn; qi++) {
        final float[] qv = qRaw[qi];
        for (int c : order) {
          QueryResult sum = runQuery(CFGS[c], shards, nSh, qv, ground[qi],
              new Random(777L * 131071L + qi));
          times[c][qi][round] = sum.ns;
          maxTimes[c][qi][round] = sum.nsMax;
          if (round == 0) {
            rec[c][qi] = sum.recall;
            dcSum[c][qi] = sum.dc;
          }
        }
      }
      System.out.printf("[sift-seg] measurement round %d/%d done%n", round + 1, ROUNDS);
    }

    // ---- aggregates ----
    double[][] med = new double[nCfg][qn];
    double[][] medMax = new double[nCfg][qn];
    for (int c = 0; c < nCfg; c++) {
      for (int qi = 0; qi < qn; qi++) {
        long[] ts = times[c][qi];
        Arrays.sort(ts);
        med[c][qi] = ts[ROUNDS / 2];
        long[] ms = maxTimes[c][qi];
        Arrays.sort(ms);
        medMax[c][qi] = ms[ROUNDS / 2];
      }
    }

    System.out.printf(Locale.ROOT, "%n==== Segmented-SIFT sweep (%d shards, n=%d, 7 rounds) ====%n",
        nSh, qn);
    System.out.printf(Locale.ROOT, "%-16s %8s %10s %10s %10s %10s%n",
        "config", "recall", "dc-sum", "CPU p50", "CPU p99", "par p50");
    double[] lats = new double[qn];
    double[] dcs = new double[qn];
    double[] recs = new double[qn];
    double[] maxs = new double[qn];
    for (int c = 0; c < nCfg; c++) {
      for (int qi = 0; qi < qn; qi++) {
        lats[qi] = med[c][qi];
        dcs[qi] = dcSum[c][qi];
        recs[qi] = rec[c][qi];
        maxs[qi] = medMax[c][qi];
      }
      Arrays.sort(lats);
      Arrays.sort(maxs);
      double rSum = 0, dSum = 0;
      for (int qi = 0; qi < qn; qi++) {
        rSum += recs[qi];
        dSum += dcs[qi];
      }
      System.out.printf(Locale.ROOT, "%-16s %8.3f %10.0f %10.3f %10.3f %10.3f%n",
          CFGS[c].name, rSum / qn, dSum / qn, lats[qn / 2] / 1e6,
          lats[(int) Math.ceil(0.99 * qn) - 1] / 1e6, maxs[qn / 2] / 1e6);
    }

    // CSV
    List<String> csv = new ArrayList<>();
    StringBuilder hdr = new StringBuilder("q");
    for (Cfg cfg : CFGS) {
      hdr.append(',').append(cfg.name).append("-rec");
    }
    for (Cfg cfg : CFGS) {
      hdr.append(',').append(cfg.name).append("-cpuNs");
    }
    for (Cfg cfg : CFGS) {
      hdr.append(',').append(cfg.name).append("-dcSum");
    }
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
        s.append(',').append(dcSum[c][qi]);
      }
      csv.add(s.toString());
    }
    try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
        outDir.resolve("phaseSIFT-seg-" + tag + ".csv"), StandardCharsets.UTF_8))) {
      for (String line : csv) {
        pw.println(line);
      }
    }
    System.out.printf("[sift-seg] wrote out/phaseSIFT-seg-%s.csv%n", tag);
  }

  // ---------------- execution ----------------

  static final class QueryResult {
    double recall;
    long dc;
    long ns;
    long nsMax;
  }

  /** One full query: every shard searched with the config's per-shard budget,
   *  shard results merged by score; ns = sum over shards (CPU), nsMax = max
   *  shard time (parallel wall estimate). */
  static QueryResult runQuery(Cfg cfg, List<Shard> shards, int nSh, float[] qv,
                              int[] ground, Random rnd) throws IOException {
    long t0 = System.nanoTime();
    List<float[]> merged = new ArrayList<>();
    long dcTotal = 0;
    long nsMax = 0;
    for (Shard sh : shards) {
      long ts0 = System.nanoTime();
      final float[][] sv = sh.vecs;
      InstrumentedSimSearch.ScorerFn scorer = ord -> -InstrumentedSearch.dist2(qv, sv[ord]);
      ShardResult sr = runShard(cfg, sh, scorer, rnd);
      long ts1 = System.nanoTime();
      dcTotal += sr.dc;
      nsMax = Math.max(nsMax, ts1 - ts0);
      for (float[] p : sr.pairs) {
        merged.add(p);
      }
    }
    merged.sort((a, b) -> Float.compare(b[1], a[1]));
    Set<Integer> seen = new HashSet<>();
    int[] top = new int[Math.min(K, merged.size())];
    int idx = 0;
    for (float[] e : merged) {
      int ord = (int) e[0];
      if (seen.add(ord)) {
        top[idx++] = ord;
        if (idx == top.length) {
          break;
        }
      }
    }
    long t1 = System.nanoTime();
    QueryResult out = new QueryResult();
    out.recall = Phase0Real.recallAt(ground, top);
    out.dc = dcTotal;
    out.ns = t1 - t0;
    out.nsMax = nsMax;
    return out;
  }

  /** Per-shard search for one config; ords are GLOBAL (shard ord + shard offset). */
  static ShardResult runShard(Cfg cfg, Shard sh, InstrumentedSimSearch.ScorerFn scorer,
                              Random rnd) throws IOException {
    ShardResult out = new ShardResult();
    int sz = sh.vecs.length;
    if (cfg.kind == 0) {
      InstrumentedSimSearch.Result r = InstrumentedSimSearch.search(
          sh.graph, sz, scorer, K, cfg.budget, null);
      out.dc = r.scoreComps;
      for (int j = 0; j < r.topK.length; j++) {
        out.pairs.add(new float[]{globalOrd(sh, r.topK[j]), r.topKScores[j]});
      }
      return out;
    }

    InstrumentedSimSearch.Result probe = InstrumentedSimSearch.search(
        sh.graph, sz, scorer, K, PROBE, null);
    out.dc = probe.scoreComps;
    for (int j = 0; j < probe.topK.length; j++) {
      out.pairs.add(new float[]{globalOrd(sh, probe.topK[j]), probe.topKScores[j]});
    }
    BitSet marked = new BitSet(sz);
    marked.or(probe.visited);
    List<float[]> union = new ArrayList<>();
    for (float[] p : out.pairs) {
      union.add(p);
    }
    for (int step = 0; step < cfg.rounds; step++) {
      int best = -1;
      if (cfg.kind == 1) {
        int[] table = sh.entryTable;
        for (int i = 0; i < table.length; i++) {
          int cand = table[(step + i) % table.length];
          if (!marked.get(cand)) {
            best = cand;
            break;
          }
        }
      } else {
        best = farthestUnvisited(sh, marked, probe.descendedNode, rnd);
      }
      if (best < 0) {
        break;
      }
      InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
          sh.graph, sz, scorer, K, cfg.perBudget, null, best, sh.nodeLevel[best], marked);
      out.dc += rr.scoreComps;
      marked.or(rr.visited);
      for (int j = 0; j < rr.topK.length; j++) {
        union.add(new float[]{globalOrd(sh, rr.topK[j]), rr.topKScores[j]});
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
    out.pairs = union;
    return out;
  }

  static int globalOrd(Shard sh, int ord) {
    return sh.offset + ord;
  }

  static int farthestUnvisited(Shard sh, BitSet marked, int from, Random rnd) {
    int best = -1;
    float bestD = -1f;
    int attempts = 0, found = 0;
    List<Integer> pool = sh.pool;
    while (found < 500 && attempts < 4000) {
      attempts++;
      int cand = pool.get(rnd.nextInt(pool.size()));
      if (marked.get(cand)) {
        continue;
      }
      found++;
      float d = InstrumentedSearch.dist2(sh.vecs[cand], sh.vecs[from]);
      if (d > bestD) {
        bestD = d;
        best = cand;
      }
    }
    return best;
  }
}
