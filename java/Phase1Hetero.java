package phase0;

import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * Heterogeneous-data test of the user's hypothesis: on a corpus with strongly
 * unbalanced clusters (one mega-cluster 80% + five small far-away clusters),
 * greedy descent always lands in the mega-cluster, so small-cluster queries are
 * STRUCTURALLY wrong-basin: no budget can cross the domain gap, only
 * restart/jump from another upper-layer entry can. Mirrors "更杂更宽泛的数据库".
 *
 * <p>Usage: Phase1Hetero
 */
public final class Phase1Hetero {

  static final int K = 10;
  static final int DIM = 32;
  static final int N_BIG = 9600;
  static final int N_SMALL_CLUSTERS = 5;
  static final int N_SMALL_PER = 480;          // N = 12000
  static final double SIGMA_BIG = 0.8;
  static final double SIGMA_SMALL = 0.15;
  static final double SEPARATION = 8.0;
  static final int M = 16;
  static final int EF_C = 100;
  static final int R = 8;                       // restarts for the oracle

  public static void main(String[] args) throws IOException {
    long seed = 42;
    int n = N_BIG + N_SMALL_CLUSTERS * N_SMALL_PER;

    // ---------- data ----------
    Random r = new Random(seed);
    float[][] vecs = new float[n][DIM];
    int[] cluster = new int[n];
    // small cluster centers on a sphere of radius SEPARATION
    float[][] centers = new float[N_SMALL_CLUSTERS][DIM];
    for (int c = 0; c < N_SMALL_CLUSTERS; c++) {
      float[] dir = new float[DIM];
      double norm = 0;
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
    for (int i = 0; i < N_BIG; i++) {           // mega cluster at origin
      for (int d = 0; d < DIM; d++) {
        vecs[idx][d] = (float) (r.nextGaussian() * SIGMA_BIG);
      }
      cluster[idx] = -1;                        // BIG
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
    // queries: 150 from mega cluster + 50 per small cluster
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

    // ---------- graph ----------
    RandomVectorScorerSupplier sup = new Phase0Main.FloatArraySupplier(vecs);
    HnswGraphBuilder b = HnswGraphBuilder.create(sup, M, EF_C, seed);
    HnswGraph graph = b.build(n);
    System.out.printf("[hetero] N=%d graph levels=%d entry=%d%n", n, graph.numLevels(), graph.entryNode());

    // upper-layer pool
    List<Integer> pool = new ArrayList<>();
    for (int level = 1; level <= graph.numLevels() - 1; level++) {
      var it = graph.getNodesOnLevel(level);
      while (it.hasNext()) {
        pool.add(it.nextInt());
      }
    }
    System.out.printf("[hetero] upper-layer pool=%d (%.1f%%)%n", pool.size(), 100.0 * pool.size() / n);

    // ---------- evaluate ----------
    Agg big = new Agg(), small = new Agg();
    for (int qi = 0; qi < qOrds.size(); qi++) {
      int qOrd = qOrds.get(qi);
      int qc = qCluster.get(qi);
      float[] q = vecs[qOrd];
      InstrumentedSimSearch.ScorerFn scorer = ord -> -InstrumentedSearch.dist2(q, vecs[ord]);
      int[] ground = bruteForce(vecs, q, K);

      double baseRec = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
          graph, n, scorer, K, 100, null).topK);
      double rec400 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
          graph, n, scorer, K, 400, null).topK);
      double rec1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
          graph, n, scorer, K, 1600, null).topK);

      // restart oracle (weak): R farthest-from-descended upper-layer entries, budget 100, best-of
      double weakCeiling = baseRec;
      // restart oracle (STRONG): closest-to-query entries, budget 400, union
      double strongCeiling = baseRec;
      if (pool.size() >= 2) {
        InstrumentedSimSearch.Result r1 = InstrumentedSimSearch.search(
            graph, n, scorer, K, 100, null);
        List<Integer> alts = farthest(pool, r1.descendedNode, vecs, R);
        for (int alt : alts) {
          double rc = Phase0Real.recallAt(ground, InstrumentedSimSearch.searchFrom(
              graph, n, scorer, K, 100, null, alt, 1).topK);
          if (rc > weakCeiling) {
            weakCeiling = rc;
          }
        }
        // strong: sample 100 pool entries, keep 4 closest to the QUERY, budget 400 each, union
        List<Integer> nearQ = closestToQuery(pool, q, vecs, 100, 4, new java.util.Random(qi * 31L));
        List<float[]> union = new ArrayList<>();
        InstrumentedSimSearch.Result rb = InstrumentedSimSearch.search(
            graph, n, scorer, K, 100, null);
        for (int i = 0; i < rb.topK.length; i++) {
          union.add(new float[]{rb.topK[i], rb.topKScores[i]});
        }
        for (int alt : nearQ) {
          InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
              graph, n, scorer, K, 400, null, alt, 1);
          for (int i = 0; i < rr.topK.length; i++) {
            union.add(new float[]{rr.topK[i], rr.topKScores[i]});
          }
        }
        strongCeiling = unionRecall(ground, union, K);
      }

      Agg a = qc < 0 ? big : small;
      a.add(baseRec, rec400, rec1600, weakCeiling, strongCeiling);
    }

    System.out.printf(Locale.ROOT, "%n==== hetero results ====%n");
    System.out.printf(Locale.ROOT,
        "MEGA-cluster queries (n=%d):  base@100=%.3f  @400=%.3f  @1600=%.3f  weakCeiling=%.3f  strongCeiling=%.3f%n",
        big.n, big.base / big.n, big.r400 / big.n, big.r1600 / big.n, big.weak / big.n,
        big.strong / big.n);
    System.out.printf(Locale.ROOT,
        "SMALL-cluster queries (n=%d): base@100=%.3f  @400=%.3f  @1600=%.3f  weakCeiling=%.3f  strongCeiling=%.3f%n",
        small.n, small.base / small.n, small.r400 / small.n, small.r1600 / small.n,
        small.weak / small.n, small.strong / small.n);
    System.out.printf(Locale.ROOT,
        "strongCeiling > single@1600: MEGA %d/%d (%.1f%%)  SMALL %d/%d (%.1f%%)%n",
        big.strongAbove, big.n, 100.0 * big.strongAbove / big.n,
        small.strongAbove, small.n, 100.0 * small.strongAbove / small.n);
  }

  static int[] bruteForce(float[][] vecs, float[] q, int k) {
    // min-heap by score: head = smallest score = farthest -> evicts farthest,
    // keeps the k nearest
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

  static List<Integer> farthest(List<Integer> pool, int from, float[][] vecs, int r) {
    Integer[] arr = pool.toArray(new Integer[0]);
    java.util.Arrays.sort(arr, (a, b) -> Float.compare(
        InstrumentedSearch.dist2(vecs[b], vecs[from]),
        InstrumentedSearch.dist2(vecs[a], vecs[from])));
    List<Integer> out = new ArrayList<>();
    for (int i = 0; i < Math.min(r, arr.length); i++) {
      out.add(arr[i]);
    }
    return out;
  }

  /** Sample pool entries, keep the r closest to the QUERY (direction-aware). */
  static List<Integer> closestToQuery(List<Integer> pool, float[] q, float[][] vecs,
                                      int sample, int r, java.util.Random rnd) {
    double[] bestD = new double[r];
    int[] bestN = new int[r];
    java.util.Arrays.fill(bestD, Double.MAX_VALUE);
    Set<Integer> seen = new HashSet<>();
    for (int i = 0; i < sample; i++) {
      int n = pool.get(rnd.nextInt(pool.size()));
      if (!seen.add(n)) {
        continue;
      }
      double d = InstrumentedSearch.dist2(vecs[n], q);
      for (int j = 0; j < r; j++) {
        if (d < bestD[j]) {
          for (int t = r - 1; t > j; t--) {
            bestD[t] = bestD[t - 1];
            bestN[t] = bestN[t - 1];
          }
          bestD[j] = d;
          bestN[j] = n;
          break;
        }
      }
    }
    List<Integer> out = new ArrayList<>();
    for (int j = 0; j < r; j++) {
      if (bestD[j] < Double.MAX_VALUE) {
        out.add(bestN[j]);
      }
    }
    return out;
  }

  /** Recall of the top-k union (by score) of all collected results. */
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

  static final class Agg {
    int n = 0;
    double base = 0, r400 = 0, r1600 = 0, weak = 0, strong = 0;
    int strongAbove = 0;

    void add(double b, double r4, double r16, double w, double s) {
      n++;
      base += b;
      r400 += r4;
      r1600 += r16;
      weak += w;
      strong += s;
      if (s > r16 + 0.01) {
        strongAbove++;
      }
    }
  }
}
