package phase0;

import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

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
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Phase 0 of Direction 1: validate H1 (bimodal separation of bad-basin queries)
 * and estimate the potential gain of the restart mechanism (H2, oracle).
 *
 * <p>Pipeline: synthetic multi-cluster Gaussian data -> Lucene 9.12 HnswGraphBuilder
 * -> brute-force ground truth -> instrumented search (faithful bytecode-verified
 * port of HnswGraphSearcher) -> classification + separability + restart oracles.
 *
 * <p>Usage: Phase0Main &lt;sigma&gt; &lt;outTag&gt; [visitLimit]
 *   visitLimit mirrors ES num_candidates (0 = unlimited).
 */
public final class Phase0Main {

  static final int DIM = 32;
  static final int NUM_CLUSTERS = 8;
  static final int PER_CLUSTER = 1500;        // N = 12000
  static final int NUM_QUERIES = 800;
  static final int K = 10;
  static final int M = 16;
  static final int EF_CONSTRUCTION = 100;
  static final long SEED = 42;
  static final double OOD_FRACTION = 0.10;

  // classification thresholds (recall@10 against ground truth)
  static final double GOOD_RECALL = 0.9;
  static final double BAD_RECALL = 0.3;

  public static void main(String[] args) throws IOException {
    if (args.length < 2) {
      System.err.println("usage: Phase0Main <sigma> <outTag> [visitLimit]");
      System.exit(2);
    }
    double sigma = Double.parseDouble(args[0]);
    String tag = args[1];
    int visitLimit = args.length >= 3 ? Integer.parseInt(args[2]) : 300;

    System.out.printf("[phase0] sigma=%.2f tag=%s visitLimit=%d%n", sigma, tag, visitLimit);

    // ---------- data ----------
    SyntheticData data = SyntheticData.generate(
        DIM, NUM_CLUSTERS, PER_CLUSTER, NUM_QUERIES, SEED, sigma, OOD_FRACTION);
    int n = data.vectors.length;
    System.out.printf("[phase0] generated N=%d dim=%d clusters=%d queries=%d%n",
        n, DIM, NUM_CLUSTERS, data.queries.length);

    // ---------- graph (real Lucene 9.12 builder, our scorer) ----------
    RandomVectorScorerSupplier supplier = new FloatArraySupplier(data.vectors);
    HnswGraphBuilder builder = HnswGraphBuilder.create(supplier, M, EF_CONSTRUCTION, SEED);
    long t0 = System.currentTimeMillis();
    HnswGraph graph = builder.build(n);   // build(maxOrd) covers ords [0, maxOrd)
    long buildMs = System.currentTimeMillis() - t0;
    if (graph.size() != n) {
      throw new IllegalStateException("graph size " + graph.size() + " != " + n);
    }
    System.out.printf("[phase0] graph built in %d ms (levels=%d, entry=%d)%n",
        buildMs, graph.numLevels(), graph.entryNode());

    // top-level node pool (for the restart oracles)
    List<Integer> topPool = new ArrayList<>();
    var it = graph.getNodesOnLevel(graph.numLevels() - 1);
    while (it.hasNext()) {
      topPool.add(it.nextInt());
    }
    System.out.printf("[phase0] top-level pool size=%d%n", topPool.size());

    // ---------- ground truth (brute force) ----------
    int[][] ground = new int[NUM_QUERIES][];
    for (int q = 0; q < NUM_QUERIES; q++) {
      ground[q] = bruteForceKnn(data.vectors, data.queries[q], K);
    }

    // ---------- instrumented search + classification + oracles ----------
    long[] D = new long[NUM_QUERIES];
    long[] patience = new long[NUM_QUERIES];
    double[] ratio = new double[NUM_QUERIES];
    double[] recall = new double[NUM_QUERIES];
    int[] visited = new int[NUM_QUERIES];
    String[] cls = new String[NUM_QUERIES];
    long[] oracleD = new long[NUM_QUERIES];          // re-search oracle
    double[] oracleRecall = new double[NUM_QUERIES];
    boolean[] oracleImproved = new boolean[NUM_QUERIES];
    long[] twoSeedD = new long[NUM_QUERIES];         // two-seed oracle
    double[] twoSeedRecall = new double[NUM_QUERIES];
    boolean[] twoSeedImproved = new boolean[NUM_QUERIES];

    long meanD = 0;
    for (int q = 0; q < NUM_QUERIES; q++) {
      InstrumentedSearch.Result r =
          InstrumentedSearch.search(data.queries[q], data.vectors, graph, K, visitLimit);
      D[q] = r.distComps;
      patience[q] = r.patience();
      ratio[q] = r.patienceRatio();
      recall[q] = recallAt(ground[q], r.topK);
      visited[q] = r.numVisited;
      meanD += r.distComps;
      cls[q] = recall[q] >= GOOD_RECALL ? "GOOD"
          : recall[q] <= BAD_RECALL ? "BAD_BASIN" : "PARTIAL";

      // oracles: only for BAD_BASIN / PARTIAL queries
      if (!cls[q].equals("GOOD")) {
        int alt = farthestFromEntry(topPool, r.entryNode, data.vectors);

        // oracle-R: full re-search with descent starting from the alternate entry
        InstrumentedSearch.Result r2 =
            InstrumentedSearch.search(data.queries[q], data.vectors, graph, K, visitLimit, alt);
        double rc2 = recallAt(ground[q], r2.topK);
        oracleD[q] = r2.distComps;
        oracleRecall[q] = rc2;
        oracleImproved[q] = rc2 > recall[q];

        // oracle-T: Lucene-native two-seed beam (eps = {descended, alt})
        InstrumentedSearch.Result r3 = InstrumentedSearch.searchTwoSeed(
            data.queries[q], data.vectors, graph, K, visitLimit, r.entryNode, alt);
        double rc3 = recallAt(ground[q], r3.topK);
        twoSeedD[q] = r3.distComps;
        twoSeedRecall[q] = rc3;
        twoSeedImproved[q] = rc3 > recall[q];
      } else {
        oracleD[q] = -1;
        oracleRecall[q] = -1;
        twoSeedD[q] = -1;
        twoSeedRecall[q] = -1;
      }
    }
    meanD /= NUM_QUERIES;

    // ---------- report ----------
    report(System.out, tag, sigma, n, visitLimit, meanD, D, patience, ratio, recall, cls,
        oracleD, oracleRecall, oracleImproved, twoSeedD, twoSeedRecall, twoSeedImproved);

    // ---------- CSV outputs ----------
    Path outDir = Paths.get("out");
    Files.createDirectories(outDir);
    try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
        outDir.resolve("per-query-" + tag + ".csv"), StandardCharsets.UTF_8))) {
      pw.println("qid,cluster,D,visited,patience,ratio,recall,class,oracleD,oracleRecall,oracleImproved,twoSeedD,twoSeedRecall,twoSeedImproved");
      for (int q = 0; q < NUM_QUERIES; q++) {
        pw.printf(Locale.ROOT, "%d,%d,%d,%d,%d,%.4f,%.3f,%s,%d,%.3f,%b,%d,%.3f,%b%n",
            q, data.queryCluster[q], D[q], visited[q], patience[q], ratio[q], recall[q],
            cls[q], oracleD[q], oracleRecall[q], oracleImproved[q],
            twoSeedD[q], twoSeedRecall[q], twoSeedImproved[q]);
      }
    }
    System.out.printf("[phase0] wrote out/per-query-%s.csv%n", tag);
  }

  // ---------- helpers ----------

  /** Brute-force top-k by squared L2. */
  static int[] bruteForceKnn(float[][] vecs, float[] q, int k) {
    PriorityQueue<Integer> pq = new PriorityQueue<>(k + 1,
        (a, b) -> Float.compare(
            InstrumentedSearch.dist2(q, vecs[b]), InstrumentedSearch.dist2(q, vecs[a])));
    for (int i = 0; i < vecs.length; i++) {
      pq.offer(i);
      if (pq.size() > k) {
        pq.poll();
      }
    }
    int[] out = new int[pq.size()];
    for (int i = out.length - 1; i >= 0; i--) {
      out[i] = pq.poll();
    }
    return out;
  }

  static double recallAt(int[] ground, int[] found) {
    Set<Integer> g = new HashSet<>(ground.length * 2);
    for (int v : ground) {
      g.add(v);
    }
    int hit = 0;
    for (int v : found) {
      if (g.contains(v)) {
        hit++;
      }
    }
    return (double) hit / ground.length;
  }

  /** Alternate entry for the restart oracles: the top-level pool node
   *  farthest (in vector space) from the current entry point. */
  static int farthestFromEntry(List<Integer> pool, int entry, float[][] vecs) {
    int best = pool.get(0);
    float bestD = -1f;
    for (int v : pool) {
      float d = InstrumentedSearch.dist2(vecs[v], vecs[entry]);
      if (d > bestD) {
        bestD = d;
        best = v;
      }
    }
    return best;
  }

  // ---------- reporting ----------

  static void report(Appendable out, String tag, double sigma, int n, int visitLimit,
                     double meanD, long[] D, long[] patience, double[] ratio, double[] recall,
                     String[] cls, long[] oracleD, double[] oracleRecall,
                     boolean[] oracleImproved, long[] twoSeedD, double[] twoSeedRecall,
                     boolean[] twoSeedImproved) throws IOException {
    int qCount = D.length;
    out.append(String.format(Locale.ROOT,
        "%n==== Phase0 report: tag=%s sigma=%.2f N=%d dim=%d clusters=%d M=%d efC=%d visitLimit=%d k=%d Q=%d ====%n",
        tag, sigma, n, DIM, NUM_CLUSTERS, M, EF_CONSTRUCTION, visitLimit, K, qCount));

    // per-class aggregates
    String[] classes = {"GOOD", "PARTIAL", "BAD_BASIN"};
    out.append(String.format("%n%-10s %6s %8s %8s %10s %10s %8s %9s%n",
        "class", "count", "meanD", "medD", "meanPat", "medPat", "meanRat", "meanRec"));
    for (String c : classes) {
      Stats sd = new Stats(), sp = new Stats(), sr = new Stats(), srec = new Stats();
      for (int q = 0; q < qCount; q++) {
        if (cls[q].equals(c)) {
          sd.add(D[q]);
          sp.add(patience[q]);
          sr.add(ratio[q]);
          srec.add(recall[q]);
        }
      }
      if (sd.count() == 0) {
        out.append(String.format("%-10s %6d (none)%n", c, 0));
      } else {
        out.append(String.format(Locale.ROOT,
            "%-10s %6d %8.0f %8.0f %10.0f %10.0f %8.3f %9.3f%n",
            c, sd.count(), sd.mean(), sd.median(), sp.mean(), sp.median(), sr.mean(), srec.mean()));
      }
    }

    // H1 separability check
    out.append(String.format(Locale.ROOT, "%n[H1 separability] meanD=%.0f%n", meanD));
    int badTotal = 0, badHit = 0, goodTotal = 0, goodHit = 0;
    for (int q = 0; q < qCount; q++) {
      if (cls[q].equals("BAD_BASIN")) {
        badTotal++;
        if (ratio[q] > 0.30 && D[q] < meanD) badHit++;
      } else if (cls[q].equals("GOOD")) {
        goodTotal++;
        if (ratio[q] < 0.10) goodHit++;
      }
    }
    out.append(String.format(Locale.ROOT,
        "  BAD_BASIN with (ratio>0.30 & D<meanD): %d/%d = %.1f%%%n", badHit, badTotal,
        badTotal == 0 ? 0 : 100.0 * badHit / badTotal));
    out.append(String.format(Locale.ROOT,
        "  GOOD      with (ratio<0.10):           %d/%d = %.1f%%%n", goodHit, goodTotal,
        goodTotal == 0 ? 0 : 100.0 * goodHit / goodTotal));

    // simple rule confusion matrix: predict BAD_BASIN if ratio>0.30 & D<meanD
    int tp = 0, fp = 0, fn = 0, tn = 0;
    for (int q = 0; q < qCount; q++) {
      boolean pred = ratio[q] > 0.30 && D[q] < meanD;
      boolean act = cls[q].equals("BAD_BASIN");
      if (pred && act) tp++;
      else if (pred) fp++;
      else if (act) fn++;
      else tn++;
    }
    out.append(String.format(Locale.ROOT,
        "%n[simple rule: predict BAD_BASIN if ratio>0.30 & D<meanD]%n"
            + "             actual BAD  actual other%n"
            + "  pred BAD   %10d  %12d%n"
            + "  pred other %10d  %12d%n"
            + "  precision=%.3f recall=%.3f%n",
        tp, fp, fn, tn,
        tp + fp == 0 ? 0 : (double) tp / (tp + fp),
        tp + fn == 0 ? 0 : (double) tp / (tp + fn)));

    // restart oracles
    reportOracle(out, "[oracle-R: full re-search from farthest top-level entry]",
        qCount, cls, recall, oracleD, oracleRecall, oracleImproved, meanD);
    reportOracle(out, "[oracle-T: two-seed beam (eps = {descended, farthest-alt})]",
        qCount, cls, recall, twoSeedD, twoSeedRecall, twoSeedImproved, meanD);

    // patience-ratio histogram by class
    out.append(String.format("%n[patience-ratio histogram]%n"));
    String[] bins = {"[0.00,0.05)", "[0.05,0.15)", "[0.15,0.30)", "[0.30,0.50)", "[0.50,1.00]"};
    out.append(String.format("%-12s %6s %8s %8s%n", "ratioBin", "GOOD", "PARTIAL", "BAD"));
    for (String b : bins) {
      int[] cnt = new int[3];
      for (int q = 0; q < qCount; q++) {
        double r = ratio[q];
        boolean in = switch (b) {
          case "[0.00,0.05)" -> r < 0.05;
          case "[0.05,0.15)" -> r >= 0.05 && r < 0.15;
          case "[0.15,0.30)" -> r >= 0.15 && r < 0.30;
          case "[0.30,0.50)" -> r >= 0.30 && r < 0.50;
          default -> r >= 0.50;
        };
        if (in) {
          int ci = cls[q].equals("GOOD") ? 0 : cls[q].equals("PARTIAL") ? 1 : 2;
          cnt[ci]++;
        }
      }
      out.append(String.format(Locale.ROOT, "%-12s %6d %8d %8d%n", b, cnt[0], cnt[1], cnt[2]));
    }
  }

  static void reportOracle(Appendable out, String label, int qCount, String[] cls,
                           double[] baseRecall, long[] oracleD, double[] oracleRecall,
                           boolean[] oracleImproved, double meanD) throws IOException {
    int improved = 0, tested = 0;
    double before = 0, after = 0;
    long cost = 0;
    for (int q = 0; q < qCount; q++) {
      if (!cls[q].equals("GOOD")) {
        tested++;
        before += baseRecall[q];
        after += oracleRecall[q];
        cost += oracleD[q];
        if (oracleImproved[q]) improved++;
      }
    }
    if (tested > 0) {
      out.append(String.format(Locale.ROOT,
          "%n%s (on non-GOOD queries only)%n"
              + "  tested=%d  improved=%d (%.1f%%)%n"
              + "  mean recall %.3f -> %.3f  (mean oracle cost %.0f dist comps, "
              + "mean base cost %.0f)%n",
          label, tested, improved, 100.0 * improved / tested,
          before / tested, after / tested, (double) cost / tested, meanD));
    }
  }

  /** Simple double-value statistics collector. */
  static final class Stats {
    private final List<Double> values = new ArrayList<>();

    void add(long v) {
      values.add((double) v);
    }

    void add(double v) {
      values.add(v);
    }

    int count() {
      return values.size();
    }

    double mean() {
      if (values.isEmpty()) return 0;
      double s = 0;
      for (double v : values) s += v;
      return s / values.size();
    }

    double median() {
      if (values.isEmpty()) return 0;
      List<Double> s = new ArrayList<>(values);
      s.sort(null);
      int m = s.size() / 2;
      return s.size() % 2 == 1 ? s.get(m) : (s.get(m - 1) + s.get(m)) / 2;
    }
  }

  /** RandomVectorScorerSupplier backed by a float[][] (squared-L2, higher = closer). */
  static final class FloatArraySupplier implements RandomVectorScorerSupplier {
    private final float[][] vecs;

    FloatArraySupplier(float[][] vecs) {
      this.vecs = vecs;
    }

    @Override
    public RandomVectorScorer scorer(int node) {
      final float[] q = vecs[node];
      return new RandomVectorScorer() {
        @Override
        public float score(int ord) {
          return -InstrumentedSearch.dist2(q, vecs[ord]);
        }

        @Override
        public int maxOrd() {
          return vecs.length - 1;
        }
      };
    }

    @Override
    public RandomVectorScorerSupplier copy() {
      return new FloatArraySupplier(vecs);
    }
  }
}
