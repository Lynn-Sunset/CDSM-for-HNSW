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
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * Final ceiling audit (回答"重启天花板怎么算的/是否被低估"):
 * STRONG ceiling = query-directed selection (pool entries closest to the
 * query, sampled) + big per-restart budget (400) + UNION of result sets.
 * Compared against single@1600 and the old weak ceiling (farthest, budget
 * 100, best-of).
 *
 * <p>Usage: Phase1CeilingAudit &lt;indexDir&gt; &lt;field&gt; &lt;numQueries&gt; &lt;tag&gt;
 */
public final class Phase1CeilingAudit {

  static final int K = 10;
  static final int SAMPLE = 100;      // pool entries scored against the query
  static final int R = 4;             // restarts kept
  static final int RESTART_BUDGET = 400;

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 4) {
      System.err.println("usage: Phase1CeilingAudit <indexDir> <field> <numQueries> <tag>");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int numQueries = Integer.parseInt(args[2]);
    String tag = args[3];

    System.out.printf("[ceiling-audit] indexDir=%s field=%s queries=%d tag=%s%n",
        indexDir, field, numQueries, tag);

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

        int[] nodeLevel = new int[size];
        java.util.Arrays.fill(nodeLevel, -1);
        List<Integer> pool = new ArrayList<>();
        for (int level = 1; level <= graph.numLevels() - 1; level++) {
          var it = graph.getNodesOnLevel(level);
          while (it.hasNext()) {
            int n = it.nextInt();
            nodeLevel[n] = level;
            pool.add(n);
          }
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
        Random rnd = new Random(314L * ctx.docBase + size);
        for (int q = 0; q < qPerLeaf; q++) {
          int qOrd = rnd.nextInt(size);
          float[] qNorm = Phase0Real.normalize(vecs[qOrd]);
          InstrumentedSimSearch.ScorerFn scorer = ord -> Phase0Real.dot(qNorm, vecs[ord]);

          InstrumentedSimSearch.Result r1 =
              InstrumentedSimSearch.search(graph, size, scorer, K, 100, accept);
          int[] ground = Phase0Real.bruteForceTopK(vecs, qNorm, K, accept);
          double baseRec = Phase0Real.recallAt(ground, r1.topK);
          if (baseRec >= 0.9) {
            continue;   // only non-good queries matter
          }
          double rec1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
              graph, size, scorer, K, 1600, accept).topK);

          // ---- weak ceiling (old method): farthest, budget 100, best-of ----
          double weak = baseRec;
          if (pool.size() >= 2) {
            List<Integer> far = farthest(pool, r1.descendedNode, vecs, Math.min(R, pool.size()));
            for (int alt : far) {
              double rc = Phase0Real.recallAt(ground, InstrumentedSimSearch.searchFrom(
                  graph, size, scorer, K, 100, accept, alt, nodeLevel[alt]).topK);
              if (rc > weak) {
                weak = rc;
              }
            }
          }

          // ---- strong ceiling: query-directed selection + budget 400 + union ----
          double strong = baseRec;
          if (pool.size() >= 2) {
            List<Integer> nearQ = closestToQuery(pool, qNorm, vecs, SAMPLE, Math.min(R, pool.size()), rnd);
            List<float[]> union = new ArrayList<>();
            for (int i = 0; i < r1.topK.length; i++) {
              union.add(new float[]{r1.topK[i], r1.topKScores[i]});
            }
            for (int alt : nearQ) {
              InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
                  graph, size, scorer, K, RESTART_BUDGET, accept, alt, nodeLevel[alt]);
              for (int i = 0; i < rr.topK.length; i++) {
                union.add(new float[]{rr.topK[i], rr.topKScores[i]});
              }
            }
            strong = unionRecall(ground, union, K);
          }

          agg.add(baseRec, rec1600, weak, strong);
        }
      }

      agg.report(System.out);
      Path outDir = Paths.get(System.getProperty("out.dir", "out"));
      Files.createDirectories(outDir);
      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
          outDir.resolve("phase1-ceiling-" + tag + ".csv"), StandardCharsets.UTF_8))) {
        pw.println("baseRec,rec1600,weakCeiling,strongCeiling");
        for (String line : agg.csvLines) {
          pw.println(line);
        }
      }
      System.out.printf("[ceiling-audit] wrote out/phase1-ceiling-%s.csv%n", tag);
    }
  }

  static List<Integer> farthest(List<Integer> pool, int from, float[][] vecs, int r) {
    Integer[] arr = pool.toArray(new Integer[0]);
    java.util.Arrays.sort(arr, (a, b) -> Float.compare(
        Phase1Jump.dist1m(vecs[b], vecs[from]), Phase1Jump.dist1m(vecs[a], vecs[from])));
    List<Integer> out = new ArrayList<>();
    for (int i = 0; i < Math.min(r, arr.length); i++) {
      out.add(arr[i]);
    }
    return out;
  }

  static List<Integer> closestToQuery(List<Integer> pool, float[] qNorm, float[][] vecs,
                                      int sample, int r, Random rnd) {
    double[] bestD = new double[r];
    int[] bestN = new int[r];
    java.util.Arrays.fill(bestD, Double.MAX_VALUE);
    Set<Integer> seen = new HashSet<>();
    for (int i = 0; i < sample; i++) {
      int n = pool.get(rnd.nextInt(pool.size()));
      if (!seen.add(n)) {
        continue;
      }
      double d = Phase1Jump.dist1m(vecs[n], qNorm);
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
    final List<String> csvLines = new ArrayList<>();
    final List<Double> baseRecs = new ArrayList<>();
    final List<Double> rec1600s = new ArrayList<>();
    final List<Double> weaks = new ArrayList<>();
    final List<Double> strongs = new ArrayList<>();

    void add(double b, double r16, double w, double s) {
      baseRecs.add(b);
      rec1600s.add(r16);
      weaks.add(w);
      strongs.add(s);
      csvLines.add(String.format(Locale.ROOT, "%.3f,%.3f,%.3f,%.3f", b, r16, w, s));
    }

    void report(Appendable out) throws IOException {
      int n = baseRecs.size();
      int strongAbove = 0, weakAbove = 0;
      for (int i = 0; i < n; i++) {
        if (strongs.get(i) > rec1600s.get(i) + 0.01) strongAbove++;
        if (weaks.get(i) > rec1600s.get(i) + 0.01) weakAbove++;
      }
      out.append(String.format(Locale.ROOT,
          "%n==== ceiling audit (non-good n=%d) ====%n", n));
      out.append(String.format(Locale.ROOT,
          "  mean: base@100=%.3f  single@1600=%.3f  weakCeiling=%.3f  strongCeiling=%.3f%n",
          mean(baseRecs), mean(rec1600s), mean(weaks), mean(strongs)));
      out.append(String.format(Locale.ROOT,
          "  ceiling > single@1600: weak %d/%d (%.1f%%)  strong %d/%d (%.1f%%)%n",
          weakAbove, n, 100.0 * weakAbove / n, strongAbove, n, 100.0 * strongAbove / n));

      // extreme tail subset
      int t = 0;
      double t16 = 0, tw = 0, ts = 0;
      int tsAbove = 0;
      for (int i = 0; i < n; i++) {
        if (rec1600s.get(i) <= 0.2) {
          t++;
          t16 += rec1600s.get(i);
          tw += weaks.get(i);
          ts += strongs.get(i);
          if (strongs.get(i) > rec1600s.get(i) + 0.01) tsAbove++;
        }
      }
      if (t > 0) {
        out.append(String.format(Locale.ROOT,
            "[extreme tail rec1600<=0.2, n=%d] single@1600=%.3f weak=%.3f strong=%.3f  strong>1600: %d/%d%n",
            t, t16 / t, tw / t, ts / t, tsAbove, t));
      }
    }

    static double mean(List<Double> xs) {
      double s = 0;
      for (double x : xs) s += x;
      return s / xs.size();
    }
  }
}
