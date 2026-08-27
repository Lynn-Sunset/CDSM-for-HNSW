package phase0;

import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Two-front jump experiment (user idea 2, refined): instead of jumping to the
 * FARTHEST node, jump to a node whose distance from the descended node equals
 * the characteristic search distance:
 *   (a) ring-R: the primary beam's actual coverage radius (measured from its
 *       visited set, 90th percentile) — per-query, edge-of-coverage seed;
 *   (b) ring-dBar: the median descended->true-nearest distance over GOOD
 *       queries (the "average distance a search needs to traverse").
 * Both variants seed a second front sharing the SAME 100-dc budget (Lucene
 * eps[] two-seed). Compared against farthest-seed two-front and single@200.
 *
 * <p>Usage: Phase1Jump &lt;indexDir&gt; &lt;field&gt; &lt;numQueries&gt; &lt;visitLimit&gt; &lt;tag&gt;
 */
public final class Phase1Jump {

  static final int K = 10;

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 5) {
      System.err.println("usage: Phase1Jump <indexDir> <field> <numQueries> <visitLimit> <tag>");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int numQueries = Integer.parseInt(args[2]);
    int visitLimit = Integer.parseInt(args[3]);
    String tag = args[4];

    System.out.printf("[phase1-jump] indexDir=%s field=%s queries=%d visitLimit=%d tag=%s%n",
        indexDir, field, numQueries, visitLimit, tag);

    try (DirectoryReader ir = DirectoryReader.open(FSDirectory.open(indexDir, NoLockFactory.INSTANCE))) {
      FSDirectory dir = (FSDirectory) ir.directory();
      List<LeafReaderContext> leaves = ir.leaves();
      Agg agg = new Agg();

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
        float[][] vecs = Phase0Real.loadVectors(fvv, size, fvv.dimension());
        HnswGraph graph = Phase0Real.getRealGraph(dir, cr, field);

        // upper-layer pool
        List<Integer> pool = new ArrayList<>();
        for (int level = 1; level <= graph.numLevels() - 1; level++) {
          var it = graph.getNodesOnLevel(level);
          while (it.hasNext()) {
            pool.add(it.nextInt());
          }
        }
        if (pool.size() < 2) {
          continue;
        }

        org.apache.lucene.util.Bits liveDocs = ctx.reader().getLiveDocs();
        RandomAccessVectorValues rav = fvv instanceof RandomAccessVectorValues r ? r : null;
        final RandomAccessVectorValues ravF = rav;
        final int sizeF = size;
        org.apache.lucene.util.Bits accept = (liveDocs != null && ravF != null)
            ? new org.apache.lucene.util.Bits() {
              @Override
              public boolean get(int ord) {
                return liveDocs.get(ravF.ordToDoc(ord));
              }

              @Override
              public int length() {
                return sizeF;
              }
            } : null;

        int qPerLeaf = Math.max(1, numQueries / leaves.size());
        long seed0 = 987L * ctx.docBase + size;

        // pass 1: calibrate dBar (median descended->true-nearest over GOOD)
        List<Double> goodDStar = new ArrayList<>();
        Random rnd1 = new Random(seed0);
        for (int q = 0; q < qPerLeaf; q++) {
          int qOrd = rnd1.nextInt(size);
          float[] qNorm = Phase0Real.normalize(vecs[qOrd]);
          InstrumentedSimSearch.ScorerFn scorer = ord -> Phase0Real.dot(qNorm, vecs[ord]);
          InstrumentedSimSearch.Result r1 =
              InstrumentedSimSearch.search(graph, size, scorer, K, visitLimit, accept);
          int[] ground = Phase0Real.bruteForceTopK(vecs, qNorm, K, accept);
          double rec = Phase0Real.recallAt(ground, r1.topK);
          if (rec >= 0.9) {
            goodDStar.add((double) dist1m(vecs[r1.descendedNode], vecs[ground[0]]));
          }
        }
        goodDStar.sort(null);
        double dBar = goodDStar.isEmpty() ? 0.0
            : goodDStar.get(goodDStar.size() / 2);
        agg.goodDStars.addAll(goodDStar);

        // pass 2: run the two-front variants on the same queries
        Random rnd2 = new Random(seed0);
        for (int q = 0; q < qPerLeaf; q++) {
          int qOrd = rnd2.nextInt(size);
          float[] qNorm = Phase0Real.normalize(vecs[qOrd]);
          InstrumentedSimSearch.ScorerFn scorer = ord -> Phase0Real.dot(qNorm, vecs[ord]);
          InstrumentedSimSearch.Result r1 =
              InstrumentedSimSearch.search(graph, size, scorer, K, visitLimit, accept);
          int[] ground = Phase0Real.bruteForceTopK(vecs, qNorm, K, accept);
          double baseRec = Phase0Real.recallAt(ground, r1.topK);
          String cls = baseRec >= 0.9 ? "GOOD" : baseRec <= 0.3 ? "BAD_BASIN" : "PARTIAL";
          double dStar = dist1m(vecs[r1.descendedNode], vecs[ground[0]]);

          if (cls.equals("GOOD")) {
            continue;   // two-front only matters for non-good
          }

          // coverage radius of the primary beam (90th percentile over visited)
          double ringR = beamRadius(r1.visited, r1.descendedNode, vecs, 0.90);
          int seedRingR = pickClosestTo(pool, r1.descendedNode, ringR, vecs, 500, rnd2);
          int seedRingD = pickClosestTo(pool, r1.descendedNode, dBar, vecs, 500, rnd2);
          int seedFar = farthestIn(pool, r1.descendedNode, vecs);
          // direction-aware jumps: the pool node closest to the QUERY itself
          // (the answer's direction) — sampling S costs S distance computations
          int seedQ50 = pickClosestToQuery(pool, qNorm, vecs, 50, rnd2);
          int seedQ500 = pickClosestToQuery(pool, qNorm, vecs, 500, rnd2);

          double twoRingR = Phase0Real.recallAt(ground, InstrumentedSimSearch.searchTwoSeed(
              graph, size, scorer, K, visitLimit, accept, graph.entryNode(), seedRingR).topK);
          double twoRingD = Phase0Real.recallAt(ground, InstrumentedSimSearch.searchTwoSeed(
              graph, size, scorer, K, visitLimit, accept, graph.entryNode(), seedRingD).topK);
          double twoFar = Phase0Real.recallAt(ground, InstrumentedSimSearch.searchTwoSeed(
              graph, size, scorer, K, visitLimit, accept, graph.entryNode(), seedFar).topK);
          double twoQ50 = Phase0Real.recallAt(ground, InstrumentedSimSearch.searchTwoSeed(
              graph, size, scorer, K, visitLimit, accept, graph.entryNode(), seedQ50).topK);
          double twoQ500 = Phase0Real.recallAt(ground, InstrumentedSimSearch.searchTwoSeed(
              graph, size, scorer, K, visitLimit, accept, graph.entryNode(), seedQ500).topK);
          double single200 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
              graph, size, scorer, K, 200, accept).topK);
          double single600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
              graph, size, scorer, K, 600, accept).topK);

          agg.add(ctx.docBase * 10000 + q, size, baseRec, cls, dStar, ringR,
              twoRingR, twoRingD, twoFar, single200, twoQ50, twoQ500, single600);
        }
      }

      agg.report(System.out);
      Path outDir = Paths.get(System.getProperty("out.dir", "out"));
      Files.createDirectories(outDir);
      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
          outDir.resolve("phase1-jump-" + tag + ".csv"), StandardCharsets.UTF_8))) {
        pw.println("qid,size,baseRec,class,dStar,ringR,twoRingR,twoRingD,twoFar,single200,twoQ50,twoQ500,single600");
        for (String line : agg.csvLines) {
          pw.println(line);
        }
      }
      System.out.printf("[phase1-jump] wrote out/phase1-jump-%s.csv%n", tag);
    }
  }

  // ---------- helpers ----------

  /** 1 - cosine over normalized stored vectors (chord-like, [0,2]). */
  static float dist1m(float[] a, float[] b) {
    return 1f - Phase0Real.dot(a, b);
  }

  /** p-th percentile of distance from {@code from} to the visited nodes. */
  static double beamRadius(BitSet visited, int from, float[][] vecs, double p) {
    List<Double> ds = new ArrayList<>();
    for (int v = visited.nextSetBit(0); v >= 0; v = visited.nextSetBit(v + 1)) {
      ds.add((double) dist1m(vecs[v], vecs[from]));
    }
    ds.sort(null);
    if (ds.isEmpty()) {
      return 0;
    }
    int idx = (int) Math.min(ds.size() - 1, p * ds.size());
    return ds.get(idx);
  }

  /** Sample pool nodes, return the one whose distance to {@code from} is
   *  closest to {@code target}. */
  static int pickClosestTo(List<Integer> pool, int from, double target, float[][] vecs,
                           int sample, Random rnd) {
    int best = pool.get(rnd.nextInt(pool.size()));
    double bestErr = Math.abs(dist1m(vecs[best], vecs[from]) - target);
    for (int i = 1; i < sample; i++) {
      int n = pool.get(rnd.nextInt(pool.size()));
      double err = Math.abs(dist1m(vecs[n], vecs[from]) - target);
      if (err < bestErr) {
        bestErr = err;
        best = n;
      }
    }
    return best;
  }

  static int farthestIn(List<Integer> pool, int from, float[][] vecs) {
    int best = pool.get(0);
    float bestD = -1f;
    for (int n : pool) {
      float d = dist1m(vecs[n], vecs[from]);
      if (d > bestD) {
        bestD = d;
        best = n;
      }
    }
    return best;
  }

  /** Direction-aware target discovery: sample {@code sample} pool nodes and
   *  return the one closest to the QUERY (the direction of the answer).
   *  Discovery cost = {@code sample} distance computations. */
  static int pickClosestToQuery(List<Integer> pool, float[] qNorm, float[][] vecs,
                                int sample, Random rnd) {
    int best = pool.get(rnd.nextInt(pool.size()));
    double bestD = dist1m(vecs[best], qNorm);
    for (int i = 1; i < sample; i++) {
      int n = pool.get(rnd.nextInt(pool.size()));
      double d = dist1m(vecs[n], qNorm);
      if (d < bestD) {
        bestD = d;
        best = n;
      }
    }
    return best;
  }

  // ---------- aggregation ----------

  static final class Agg {
    final List<String> csvLines = new ArrayList<>();
    final List<Double> goodDStars = new ArrayList<>();
    final List<Double> badDStars = new ArrayList<>();
    final List<Double> baseRecs = new ArrayList<>();
    final List<Double> ringRs = new ArrayList<>();
    final List<Double> twoRingRs = new ArrayList<>();
    final List<Double> twoRingDs = new ArrayList<>();
    final List<Double> twoFars = new ArrayList<>();
    final List<Double> single200s = new ArrayList<>();
    final List<Double> q50s = new ArrayList<>();
    final List<Double> q500s = new ArrayList<>();
    final List<Double> single600s = new ArrayList<>();

    void add(int qid, int size, double baseRec, String cls, double dStar, double ringR,
             double twoRingR, double twoRingD, double twoFar, double single200,
             double twoQ50, double twoQ500, double single600) {
      badDStars.add(dStar);
      baseRecs.add(baseRec);
      ringRs.add(ringR);
      twoRingRs.add(twoRingR);
      twoRingDs.add(twoRingD);
      twoFars.add(twoFar);
      single200s.add(single200);
      q50s.add(twoQ50);
      q500s.add(twoQ500);
      single600s.add(single600);
      csvLines.add(String.format(Locale.ROOT,
          "%d,%d,%.3f,%s,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
          qid, size, baseRec, cls, dStar, ringR, twoRingR, twoRingD, twoFar,
          single200, twoQ50, twoQ500, single600));
    }

    void report(Appendable out) throws IOException {
      int n = baseRecs.size();
      out.append(String.format(Locale.ROOT,
          "%n==== Phase1-jump report (non-good n=%d) ====%n", n));
      out.append(String.format(Locale.ROOT,
          "[calibration] GOOD: median dStar=%.4f p90=%.4f (n=%d)%n",
          median(goodDStars), pct(goodDStars, 0.90), goodDStars.size()));
      out.append(String.format(Locale.ROOT,
          "[bad queries] median dStar=%.4f p90=%.4f  median beamRingR=%.4f%n",
          median(badDStars), pct(badDStars, 0.90), median(ringRs)));
      out.append(String.format(Locale.ROOT,
          "%n[two-front variants, shared budget 100]%n"
              + "  single-seed base      = %.3f%n"
              + "  ring-R seed (coverage edge) = %.3f%n"
              + "  ring-dBar seed (calibrated) = %.3f%n"
              + "  farthest seed         = %.3f%n"
              + "  single @200           = %.3f%n",
          mean(baseRecs), mean(twoRingRs), mean(twoRingDs), mean(twoFars), mean(single200s)));
      out.append(String.format(Locale.ROOT,
          "%n[direction-aware jumps: pool node closest to the QUERY]%n"
              + "  jump-S50  (cost ~150 dc) = %.3f   vs single@200 = %.3f%n"
              + "  jump-S500 (cost ~600 dc) = %.3f   vs single@600 = %.3f%n",
          mean(q50s), mean(single200s), mean(q500s), mean(single600s)));
      out.append(String.format(Locale.ROOT,
          "improvement fraction vs base: ring-R %.1f%%  ring-dBar %.1f%%  farthest %.1f%%%n",
          100.0 * frac(twoRingRs, baseRecs), 100.0 * frac(twoRingDs, baseRecs),
          100.0 * frac(twoFars, baseRecs)));
    }

    static double frac(List<Double> a, List<Double> b) {
      int c = 0;
      for (int i = 0; i < a.size(); i++) {
        if (a.get(i) > b.get(i)) c++;
      }
      return (double) c / a.size();
    }

    static double mean(List<Double> xs) {
      double s = 0;
      for (double x : xs) s += x;
      return s / xs.size();
    }

    static double median(List<Double> xs) {
      if (xs.isEmpty()) return 0;
      List<Double> s = new ArrayList<>(xs);
      s.sort(null);
      return s.get(s.size() / 2);
    }

    static double pct(List<Double> xs, double p) {
      if (xs.isEmpty()) return 0;
      List<Double> s = new ArrayList<>(xs);
      s.sort(null);
      return s.get((int) Math.min(s.size() - 1, p * s.size()));
    }
  }
}
