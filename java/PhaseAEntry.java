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
 * Phase A (PVLDB plan): entry DISCOVERY structures — the cost of finding
 * query-directed multi-entry seeds vs their quality.
 *
 * <p>Discovery-cost ladder (per query, entries all evaluated identically:
 * 4 entries x budget-400 restarts, union with the base top-k):
 *   0 dc  : fixed 4 random pool entries (precomputed)
 *   C dc  : k-means over the upper-layer pool with C in {16,32,64} centers;
 *           query scores the C centers, takes the cell's precomputed top-4
 *   100 dc: sampling oracle (100 pool entries scored against the query)
 * Reference: single@1600.
 *
 * <p>Goal: the <=20 dc rung matches the 100-dc oracle within noise.
 *
 * <p>Usage: PhaseAEntry &lt;indexDir&gt; &lt;field&gt; &lt;numQueries&gt; &lt;tag&gt;
 */
public final class PhaseAEntry {

  static final int K = 10;
  static final int R_ENTRIES = 4;
  static final int RESTART_BUDGET = 400;

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 4) {
      System.err.println("usage: PhaseAEntry <indexDir> <field> <numQueries> <tag>");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int numQueries = Integer.parseInt(args[2]);
    String tag = args[3];

    System.out.printf("[phaseA] indexDir=%s field=%s queries=%d tag=%s%n",
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
        if (pool.size() < 4) {
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

        // ---- build-time structures (per leaf) ----
        long seedB = 777L * ctx.docBase + size;
        int[] random4 = randomEntries(pool, R_ENTRIES, new Random(seedB));
        Cells[] cellStructures = new Cells[3];
        int[] cs = {16, 32, 64};
        for (int ci = 0; ci < cs.length; ci++) {
          cellStructures[ci] = kmeansCells(pool, vecs, cs[ci], new Random(seedB + ci));
        }

        int qPerLeaf = Math.max(1, numQueries / leaves.size());
        Random rnd = new Random(4242L * ctx.docBase + size);
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
          double rec1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
              graph, size, scorer, K, 1600, accept).topK);

          List<float[]> unionBase = new ArrayList<>();
          for (int i = 0; i < r1.topK.length; i++) {
            unionBase.add(new float[]{r1.topK[i], r1.topKScores[i]});
          }

          // ladder rungs
          double r0 = runEntries(graph, size, scorer, accept, nodeLevel, random4,
              ground, unionBase);
          double r16 = 0, r32 = 0, r64 = 0;
          for (int ci = 0; ci < cs.length; ci++) {
            int cell = argminCenter(cellStructures[ci].centers, cs[ci], qNorm);
            int[] entries = new int[R_ENTRIES];
            for (int j = 0; j < R_ENTRIES; j++) {
              entries[j] = cellStructures[ci].entryTable[cell * R_ENTRIES + j];
            }
            double rec = runEntries(graph, size, scorer, accept, nodeLevel, entries,
                ground, unionBase);
            if (ci == 0) r16 = rec;
            else if (ci == 1) r32 = rec;
            else r64 = rec;
          }
          // 100-dc sampling oracle
          List<Integer> nearQ = top4Closest(pool, qNorm, vecs, 100, rnd);
          int[] nearQArr = new int[nearQ.size()];
          for (int i = 0; i < nearQArr.length; i++) {
            nearQArr[i] = nearQ.get(i);
          }
          double r100 = runEntries(graph, size, scorer, accept, nodeLevel, nearQArr,
              ground, unionBase);

          agg.add(baseRec, rec1600, r0, r16, r32, r64, r100);
        }
      }

      agg.report(System.out);
      Path outDir = Paths.get(System.getProperty("out.dir", "out"));
      Files.createDirectories(outDir);
      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
          outDir.resolve("phaseA-" + tag + ".csv"), StandardCharsets.UTF_8))) {
        pw.println("baseRec,rec1600,r0,r16,r32,r64,r100");
        for (String line : agg.csvLines) {
          pw.println(line);
        }
      }
      System.out.printf("[phaseA] wrote out/phaseA-%s.csv%n", tag);
    }
  }

  /** Evaluate a fixed entry set: 4 restarts (budget 400), union with base. */
  static double runEntries(HnswGraph graph, int size, InstrumentedSimSearch.ScorerFn scorer,
                           org.apache.lucene.util.Bits accept, int[] nodeLevel,
                           int[] entries, int[] ground, List<float[]> unionBase)
      throws IOException {
    List<float[]> union = new ArrayList<>(unionBase);
    for (int e : entries) {
      if (e < 0 || nodeLevel[e] < 0) {
        continue;
      }
      InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
          graph, size, scorer, K, RESTART_BUDGET, accept, e, nodeLevel[e]);
      for (int i = 0; i < rr.topK.length; i++) {
        union.add(new float[]{rr.topK[i], rr.topKScores[i]});
      }
    }
    return unionRecall(ground, union, K);
  }

  /** Build-time cell structure: k-means centers + per-cell top entry table. */
  static final class Cells {
    final float[][] centers;
    final int[] entryTable;

    Cells(float[][] centers, int[] entryTable) {
      this.centers = centers;
      this.entryTable = entryTable;
    }
  }

  static int[] randomEntries(List<Integer> pool, int r, Random rnd) {
    Set<Integer> seen = new HashSet<>();
    int[] out = new int[r];
    int i = 0;
    while (i < r) {
      int n = pool.get(rnd.nextInt(pool.size()));
      if (seen.add(n)) {
        out[i++] = n;
      }
    }
    return out;
  }

  /** K-means over the pool with C centers; returns centers + per-cell entry table
   *  (entryTable[cell*R + j] = j-th closest pool node to that center). */
  static Cells kmeansCells(List<Integer> pool, float[][] vecs, int C, Random rnd) {
    int p = pool.size();
    // k-means++ init
    float[][] centers = new float[C][];
    centers[0] = vecs[pool.get(rnd.nextInt(p))].clone();
    double[] minD2 = new double[p];
    for (int i = 0; i < p; i++) {
      minD2[i] = InstrumentedSearch.dist2(vecs[pool.get(i)], centers[0]);
    }
    for (int c = 1; c < C; c++) {
      double total = 0;
      for (double d : minD2) {
        total += d;
      }
      double t = rnd.nextDouble() * total;
      int pick = p - 1;
      for (int i = 0; i < p; i++) {
        t -= minD2[i];
        if (t <= 0) {
          pick = i;
          break;
        }
      }
      centers[c] = vecs[pool.get(pick)].clone();
      for (int i = 0; i < p; i++) {
        double d = InstrumentedSearch.dist2(vecs[pool.get(i)], centers[c]);
        if (d < minD2[i]) {
          minD2[i] = d;
        }
      }
    }
    int[] assign = new int[p];
    int[] entryTable = new int[C * R_ENTRIES];
    float[] bestD = new float[C * R_ENTRIES];
    java.util.Arrays.fill(bestD, Float.MAX_VALUE);
    int[] cellCounts = new int[C];
    for (int iter = 0; iter < 15; iter++) {
      double[][] sums = new double[C][vecs[0].length];
      java.util.Arrays.fill(entryTable, -1);
      java.util.Arrays.fill(bestD, Float.MAX_VALUE);
      java.util.Arrays.fill(cellCounts, 0);
      for (int i = 0; i < p; i++) {
        float[] v = vecs[pool.get(i)];
        int best = 0;
        float bd = Float.MAX_VALUE;
        for (int c = 0; c < C; c++) {
          float d = InstrumentedSearch.dist2(v, centers[c]);
          if (d < bd) {
            bd = d;
            best = c;
          }
        }
        assign[i] = best;
        cellCounts[best]++;
        double[] s = sums[best];
        for (int d = 0; d < v.length; d++) {
          s[d] += v[d];
        }
        // maintain per-cell top-4 by distance to the CURRENT center
        int base = best * R_ENTRIES;
        for (int j = 0; j < R_ENTRIES; j++) {
          if (bd < bestD[base + j]) {
            for (int t = R_ENTRIES - 1; t > j; t--) {
              bestD[base + t] = bestD[base + t - 1];
              entryTable[base + t] = entryTable[base + t - 1];
            }
            bestD[base + j] = bd;
            entryTable[base + j] = pool.get(i);
            break;
          }
        }
      }
      for (int c = 0; c < C; c++) {
        if (cellCounts[c] == 0) {
          centers[c] = vecs[pool.get(rnd.nextInt(p))].clone();
          continue;
        }
        float[] m = centers[c];
        double[] s = sums[c];
        for (int d = 0; d < m.length; d++) {
          m[d] = (float) (s[d] / cellCounts[c]);
        }
      }
    }
    // last-iteration entries use distances to the PRE-update centers; acceptable
    // for a build-time structure (one extra assignment pass for correctness):
    for (int i = 0; i < p; i++) {
      float[] v = vecs[pool.get(i)];
      int best = 0;
      float bd = Float.MAX_VALUE;
      for (int c = 0; c < C; c++) {
        float d = InstrumentedSearch.dist2(v, centers[c]);
        if (d < bd) {
          bd = d;
          best = c;
        }
      }
      int base = best * R_ENTRIES;
      for (int j = 0; j < R_ENTRIES; j++) {
        if (bd < bestD[base + j]) {
          for (int t = R_ENTRIES - 1; t > j; t--) {
            bestD[base + t] = bestD[base + t - 1];
            entryTable[base + t] = entryTable[base + t - 1];
          }
          bestD[base + j] = bd;
          entryTable[base + j] = pool.get(i);
          break;
        }
      }
    }
    // fill empty cell slots with random pool entries
    for (int c = 0; c < C; c++) {
      for (int j = 0; j < R_ENTRIES; j++) {
        if (entryTable[c * R_ENTRIES + j] < 0) {
          entryTable[c * R_ENTRIES + j] = pool.get(rnd.nextInt(p));
        }
      }
    }
    return new Cells(centers, entryTable);
  }

  static int argminCenter(float[][] centers, int C, float[] qNorm) {
    int best = 0;
    float bd = Float.MAX_VALUE;
    for (int c = 0; c < C; c++) {
      float d = InstrumentedSearch.dist2(qNorm, centers[c]);
      if (d < bd) {
        bd = d;
        best = c;
      }
    }
    return best;
  }

  static List<Integer> top4Closest(List<Integer> pool, float[] qNorm, float[][] vecs,
                                   int sample, Random rnd) {
    float[] bestD = new float[R_ENTRIES];
    int[] bestN = new int[R_ENTRIES];
    java.util.Arrays.fill(bestD, Float.MAX_VALUE);
    Set<Integer> seen = new HashSet<>();
    for (int i = 0; i < sample; i++) {
      int n = pool.get(rnd.nextInt(pool.size()));
      if (!seen.add(n)) {
        continue;
      }
      float d = InstrumentedSearch.dist2(qNorm, vecs[n]);
      for (int j = 0; j < R_ENTRIES; j++) {
        if (d < bestD[j]) {
          for (int t = R_ENTRIES - 1; t > j; t--) {
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
    for (int j = 0; j < R_ENTRIES; j++) {
      if (bestN[j] != 0 || bestD[j] < Float.MAX_VALUE) {
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
    final List<List<Double>> ladder = List.of(
        new ArrayList<>(), new ArrayList<>(), new ArrayList<>(),
        new ArrayList<>(), new ArrayList<>());
    int n = 0;

    void add(double b, double r16, double r0, double r16k, double r32, double r64, double r100) {
      n++;
      baseRecs.add(b);
      rec1600s.add(r16);
      ladder.get(0).add(r0);
      ladder.get(1).add(r16k);
      ladder.get(2).add(r32);
      ladder.get(3).add(r64);
      ladder.get(4).add(r100);
      csvLines.add(String.format(Locale.ROOT, "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
          b, r16, r0, r16k, r32, r64, r100));
    }

    void report(Appendable out) throws IOException {
      String[] names = {"0(random4)", "16(kmeans)", "32(kmeans)", "64(kmeans)", "100(oracle)"};
      out.append(String.format(Locale.ROOT,
          "%n==== PhaseA discovery-cost ladder (non-good n=%d) ====%n", n));
      out.append(String.format(Locale.ROOT,
          "  reference: base@100=%.3f  single@1600=%.3f%n", mean(baseRecs), mean(rec1600s)));
      for (int i = 0; i < ladder.size(); i++) {
        int above = 0;
        for (int j = 0; j < n; j++) {
          if (ladder.get(i).get(j) > rec1600s.get(j) + 0.01) {
            above++;
          }
        }
        out.append(String.format(Locale.ROOT,
            "  %-12s dc: recall=%.3f   plateauBreak %d/%d (%.1f%%)%n",
            names[i], mean(ladder.get(i)), above, n, 100.0 * above / n));
      }
    }

    static double mean(List<Double> xs) {
      double s = 0;
      for (double x : xs) s += x;
      return s / xs.size();
    }
  }
}
