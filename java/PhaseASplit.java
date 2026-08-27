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
 * Phase A / RQ3: budget SPLIT sweep for the random multi-entry union.
 * Iso-total-cost (~1600 dc) comparison: single@1600 vs R independent beams
 * with per-beam budget 1600/R, union of results, R in {2,4,8,16}.
 * Entries: fixed random upper-layer nodes (0 discovery cost, per Phase A).
 *
 * <p>Usage: PhaseASplit &lt;indexDir&gt; &lt;field&gt; &lt;numQueries&gt; &lt;tag&gt;
 */
public final class PhaseASplit {

  static final int K = 10;

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 4) {
      System.err.println("usage: PhaseASplit <indexDir> <field> <numQueries> <tag>");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int numQueries = Integer.parseInt(args[2]);
    String tag = args[3];

    System.out.printf("[phaseA-split] indexDir=%s field=%s queries=%d tag=%s%n",
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
        if (pool.size() < 16) {
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

        // fixed random 16 entries (0 discovery cost, per Phase A finding)
        Random rndE = new Random(918L * ctx.docBase + size);
        Set<Integer> seen = new HashSet<>();
        int[] entries16 = new int[16];
        int ei = 0;
        while (ei < 16) {
          int n = pool.get(rndE.nextInt(pool.size()));
          if (seen.add(n)) {
            entries16[ei++] = n;
          }
        }

        int qPerLeaf = Math.max(1, numQueries / leaves.size());
        Random rnd = new Random(1357L * ctx.docBase + size);
        for (int q = 0; q < qPerLeaf; q++) {
          int qOrd = rnd.nextInt(size);
          float[] qNorm = Phase0Real.normalize(vecs[qOrd]);
          InstrumentedSimSearch.ScorerFn scorer = ord -> Phase0Real.dot(qNorm, vecs[ord]);

          InstrumentedSimSearch.Result r1 =
              InstrumentedSimSearch.search(graph, size, scorer, K, 100, accept);
          int[] ground = Phase0Real.bruteForceTopK(vecs, qNorm, K, accept);
          double baseRec = Phase0Real.recallAt(ground, r1.topK);
          if (baseRec >= 0.9) {
            continue;
          }
          // trajectory signals from the base search (Phase C trigger features)
          int impCount = r1.improveTrace.length;
          int lateImps = 0;
          for (int step : r1.improveTrace) {
            if (step > 0.8 * r1.scoreComps) {
              lateImps++;
            }
          }
          double lateFrac = impCount == 0 ? 0 : (double) lateImps / impCount;
          double patienceRatio = r1.patienceRatio();
          double foundBest = r1.bestScore();
          double single1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
              graph, size, scorer, K, 1600, accept).topK);
          double single400 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
              graph, size, scorer, K, 400, accept).topK);
          double single800 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
              graph, size, scorer, K, 800, accept).topK);

          List<float[]> unionBase = new ArrayList<>();
          for (int i = 0; i < r1.topK.length; i++) {
            unionBase.add(new float[]{r1.topK[i], r1.topKScores[i]});
          }

          double[] splitRecs = new double[4];
          int[] rs = {2, 4, 8, 16};
          for (int si = 0; si < rs.length; si++) {
            int R = rs[si];
            int perBeam = 1600 / R;
            List<float[]> union = new ArrayList<>(unionBase);
            for (int i = 0; i < R; i++) {
              int e = entries16[i];
              InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
                  graph, size, scorer, K, perBeam, accept, e, nodeLevel[e]);
              for (int j = 0; j < rr.topK.length; j++) {
                union.add(new float[]{rr.topK[j], rr.topKScores[j]});
              }
            }
            splitRecs[si] = unionRecall(ground, union, K);
          }

          agg.add(baseRec, single1600, splitRecs, impCount, lateFrac, patienceRatio, foundBest,
              single400, single800);
        }
      }

      agg.report(System.out);
      Path outDir = Paths.get(System.getProperty("out.dir", "out"));
      Files.createDirectories(outDir);
      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
          outDir.resolve("phaseA-split-" + tag + ".csv"), StandardCharsets.UTF_8))) {
        pw.println("baseRec,single1600,r2x800,r4x400,r8x200,r16x100,impCount,lateFrac,patienceRatio,foundBest,single400,single800");
        for (String line : agg.csvLines) {
          pw.println(line);
        }
      }
      System.out.printf("[phaseA-split] wrote out/phaseA-split-%s.csv%n", tag);
    }
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
    final List<Double> single1600s = new ArrayList<>();
    final List<List<Double>> splits = List.of(
        new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
    final List<Double> impCounts = new ArrayList<>();
    final List<Double> lateFracs = new ArrayList<>();
    final List<Double> patiences = new ArrayList<>();
    final List<Double> foundBests = new ArrayList<>();
    final List<Double> s400s = new ArrayList<>();
    final List<Double> s800s = new ArrayList<>();
    int n = 0;

    void add(double b, double s16, double[] recs, int impCount, double lateFrac,
             double patienceRatio, double foundBest, double s400, double s800) {
      n++;
      baseRecs.add(b);
      single1600s.add(s16);
      for (int i = 0; i < recs.length; i++) {
        splits.get(i).add(recs[i]);
      }
      impCounts.add((double) impCount);
      lateFracs.add(lateFrac);
      patiences.add(patienceRatio);
      foundBests.add(foundBest);
      s400s.add(s400);
      s800s.add(s800);
      csvLines.add(String.format(Locale.ROOT,
          "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.3f,%.3f,%.4f,%.3f,%.3f",
          b, s16, recs[0], recs[1], recs[2], recs[3], impCount, lateFrac,
          patienceRatio, foundBest, s400, s800));
    }

    void report(Appendable out) throws IOException {
      String[] names = {"2x800", "4x400", "8x200", "16x100"};
      out.append(String.format(Locale.ROOT,
          "%n==== PhaseA split sweep (non-good n=%d, iso-cost ~1600 dc) ====%n", n));
      out.append(String.format(Locale.ROOT,
          "  reference: base@100=%.3f  single@1600=%.3f%n", mean(baseRecs), mean(single1600s)));
      for (int i = 0; i < splits.size(); i++) {
        int above = 0;
        for (int j = 0; j < n; j++) {
          if (splits.get(i).get(j) > single1600s.get(j) + 0.01) {
            above++;
          }
        }
        out.append(String.format(Locale.ROOT,
            "  %-7s : recall=%.3f (delta vs single=%+.3f)  plateauBreak %d/%d (%.1f%%)%n",
            names[i], mean(splits.get(i)), mean(splits.get(i)) - mean(single1600s),
            above, n, 100.0 * above / n));
      }
    }

    static double mean(List<Double> xs) {
      double s = 0;
      for (double x : xs) s += x;
      return s / xs.size();
    }
  }
}
