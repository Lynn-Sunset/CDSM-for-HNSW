package phase0;

import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * Phase A / RQ3 on HETEROGENEOUS synthetic data: unbalanced clusters
 * (mega 80% + five small far-away). Verifies whether the multi-entry union's
 * plateau-break grows with heterogeneity (predicted by Phase 1 evidence).
 *
 * <p>Usage: PhaseAHetero
 */
public final class PhaseAHetero {

  static final int K = 10;
  static final int DIM = 32;
  static final int N_BIG = 9600;
  static final int N_SMALL_CLUSTERS = 5;
  static final int N_SMALL_PER = 480;
  static final double SIGMA_BIG = 0.8;
  static final double SIGMA_SMALL = 0.15;
  static final double SEPARATION = 8.0;

  public static void main(String[] args) throws IOException {
    long seed = 42;
    int n = N_BIG + N_SMALL_CLUSTERS * N_SMALL_PER;

    Random r = new Random(seed);
    float[][] vecs = new float[n][DIM];
    int[] cluster = new int[n];
    float[][] centers = new float[N_SMALL_CLUSTERS][DIM];
    for (int c = 0; c < N_SMALL_CLUSTERS; c++) {
      double norm = 0;
      float[] dir = new float[DIM];
      for (int d = 0; d < DIM; d++) {
        dir[d] = (float) r.nextGaussian();
        norm += dir[d] * dir[d];
      }
      float s = (float) (SEPARATION / Math.sqrt(norm));
      for (int d = 0; d < DIM; d++) {
        centers[c][d] = dir[d] * s;
      }
    }
    int idx = 0;
    for (int i = 0; i < N_BIG; i++) {
      for (int d = 0; d < DIM; d++) {
        vecs[idx][d] = (float) (r.nextGaussian() * SIGMA_BIG);
      }
      cluster[idx] = -1;
      idx++;
    }
    for (int c = 0; c < N_SMALL_CLUSTERS; c++) {
      for (int i = 0; i < N_SMALL_PER; i++) {
        for (int d = 0; d < DIM; d++) {
          vecs[idx][d] = centers[c][d] + (float) (r.nextGaussian() * SIGMA_SMALL);
        }
        cluster[idx] = c;
        idx++;
      }
    }

    List<Integer> qOrds = new ArrayList<>();
    List<Integer> qCluster = new ArrayList<>();
    Random qr = new Random(7);
    for (int i = 0; i < 150; i++) {
      qOrds.add(qr.nextInt(N_BIG));
      qCluster.add(-1);
    }
    for (int c = 0; c < N_SMALL_CLUSTERS; c++) {
      for (int i = 0; i < 50; i++) {
        qOrds.add(N_BIG + c * N_SMALL_PER + qr.nextInt(N_SMALL_PER));
        qCluster.add(c);
      }
    }

    RandomVectorScorerSupplier sup = new Phase0Main.FloatArraySupplier(vecs);
    HnswGraph graph = HnswGraphBuilder.create(sup, 16, 100, seed).build(n);

    int[] nodeLevel = new int[n];
    java.util.Arrays.fill(nodeLevel, -1);
    List<Integer> pool = new ArrayList<>();
    for (int level = 1; level <= graph.numLevels() - 1; level++) {
      var it = graph.getNodesOnLevel(level);
      while (it.hasNext()) {
        int node = it.nextInt();
        nodeLevel[node] = level;
        pool.add(node);
      }
    }
    Random rndE = new Random(918);
    Set<Integer> seen = new HashSet<>();
    int[] entries16 = new int[16];
    int ei = 0;
    while (ei < 16) {
      int node = pool.get(rndE.nextInt(pool.size()));
      if (seen.add(node)) {
        entries16[ei++] = node;
      }
    }

    Agg big = new Agg(), small = new Agg();
    List<String> csvLines = new ArrayList<>();
    for (int qi = 0; qi < qOrds.size(); qi++) {
      int qOrd = qOrds.get(qi);
      float[] q = vecs[qOrd];
      InstrumentedSimSearch.ScorerFn scorer = ord -> -InstrumentedSearch.dist2(q, vecs[ord]);
      int[] ground = bruteForce(vecs, q, K);
      double baseRec = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
          graph, n, scorer, K, 100, null).topK);
      if (baseRec >= 0.9) {
        continue;
      }
      double single1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
          graph, n, scorer, K, 1600, null).topK);

      List<float[]> unionBase = new ArrayList<>();
      InstrumentedSimSearch.Result rb = InstrumentedSimSearch.search(
          graph, n, scorer, K, 100, null);
      for (int i = 0; i < rb.topK.length; i++) {
        unionBase.add(new float[]{rb.topK[i], rb.topKScores[i]});
      }

      double two800 = splitUnion(graph, n, scorer, ground, unionBase, entries16,
          nodeLevel, 2, 800);
      double four400 = splitUnion(graph, n, scorer, ground, unionBase, entries16,
          nodeLevel, 4, 400);
      double rec200 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
          graph, n, scorer, K, 200, null).topK);
      double rec400 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
          graph, n, scorer, K, 400, null).topK);
      double rec800 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
          graph, n, scorer, K, 800, null).topK);

      Agg a = qCluster.get(qi) < 0 ? big : small;
      a.add(baseRec, rec200, rec400, rec800, single1600, two800, four400);
      csvLines.add(String.format(Locale.ROOT, "%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
          qCluster.get(qi), baseRec, rec200, rec400, rec800, single1600, two800, four400));
    }

    System.out.printf(Locale.ROOT, "%n==== PhaseA hetero sweep ====%n");
    System.out.printf(Locale.ROOT,
        "MEGA (n=%d):  base@100=%.3f  @200=%.3f  @400=%.3f  @800=%.3f  @1600=%.3f  2x800=%.3f  4x400=%.3f  (breaks: %d/%d, %d/%d)%n",
        big.n, big.base / big.n, big.r200 / big.n, big.r400 / big.n, big.r800 / big.n,
        big.s16 / big.n, big.r2 / big.n, big.r4 / big.n, big.b2, big.n, big.b4, big.n);
    System.out.printf(Locale.ROOT,
        "SMALL(n=%d):  base@100=%.3f  @200=%.3f  @400=%.3f  @800=%.3f  @1600=%.3f  2x800=%.3f  4x400=%.3f  (breaks: %d/%d, %d/%d)%n",
        small.n, small.base / small.n, small.r200 / small.n, small.r400 / small.n,
        small.r800 / small.n, small.s16 / small.n, small.r2 / small.n, small.r4 / small.n,
        small.b2, small.n, small.b4, small.n);

    // per-query CSV
    java.nio.file.Path outDir = java.nio.file.Paths.get(System.getProperty("out.dir", "out"));
    java.nio.file.Files.createDirectories(outDir);
    try (java.io.PrintWriter pw = new java.io.PrintWriter(java.nio.file.Files.newBufferedWriter(
        outDir.resolve("phaseA-hetero.csv"), java.nio.charset.StandardCharsets.UTF_8))) {
      pw.println("clusterType,baseRec,rec200,rec400,rec800,single1600,two800,four400");
      for (String line : csvLines) {
        pw.println(line);
      }
    }
    System.out.println("[phaseA-hetero] wrote out/phaseA-hetero.csv");
  }

  static double splitUnion(HnswGraph graph, int size, InstrumentedSimSearch.ScorerFn scorer,
                           int[] ground, List<float[]> unionBase, int[] entries,
                           int[] nodeLevel, int R, int perBeam) throws IOException {
    List<float[]> union = new ArrayList<>(unionBase);
    for (int i = 0; i < R; i++) {
      InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
          graph, size, scorer, K, perBeam, null, entries[i], nodeLevel[entries[i]]);
      for (int j = 0; j < rr.topK.length; j++) {
        union.add(new float[]{rr.topK[j], rr.topKScores[j]});
      }
    }
    List<float[]> sorted = new ArrayList<>(union);
    sorted.sort((a, b) -> Float.compare(b[1], a[1]));
    Set<Integer> seen = new HashSet<>();
    int[] top = new int[Math.min(K, sorted.size())];
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

  static int[] bruteForce(float[][] vecs, float[] q, int k) {
    java.util.PriorityQueue<Integer> pq = new java.util.PriorityQueue<>(k + 1,
        (a, b) -> Float.compare(
            -InstrumentedSearch.dist2(q, vecs[a]), -InstrumentedSearch.dist2(q, vecs[b])));
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

  static final class Agg {
    int n = 0;
    double base = 0, r200 = 0, r400 = 0, r800 = 0, s16 = 0, r2 = 0, r4 = 0;
    int b2 = 0, b4 = 0;

    void add(double b, double r2_, double r4_, double r8, double s, double two, double four) {
      n++;
      base += b;
      r200 += r2_;
      r400 += r4_;
      r800 += r8;
      s16 += s;
      r2 += two;
      r4 += four;
      if (two > s + 0.01) {
        b2++;
      }
      if (four > s + 0.01) {
        b4++;
      }
    }
  }
}
