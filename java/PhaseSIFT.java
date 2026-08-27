package phase0;

import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
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
 * Homogeneous classic control: SIFT1M (128-d, L2, standard fvecs/ivecs).
 * Loads sift_base.fvecs / sift_query.fvecs / sift_groundtruth.ivecs,
 * builds a real HNSW (L2 scorer, M=16 efC=100) and runs the fair
 * four-strategy comparison (per-entry budget 400 + union) vs single@400/@1600.
 *
 * <p>Usage: PhaseSIFT &lt;dataDir&gt; &lt;numQueries&gt; &lt;tag&gt;
 */
public final class PhaseSIFT {

  static final int K = 10;
  static final int M = 16;
  static final int EF_C = 100;
  static final int R = 4;
  static final int BUDGET = 400;

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 3) {
      System.err.println("usage: PhaseSIFT <dataDir> <numQueries> <tag>");
      System.exit(2);
    }
    Path dir = Paths.get(args[0]);
    int numQueries = Integer.parseInt(args[1]);
    String tag = args[2];

    System.out.println("[sift] loading sift_base.fvecs ...");
    float[][] base = loadFvecs(dir.resolve("sift_base.fvecs"));
    System.out.printf("[sift] base: %d x %d%n", base.length, base[0].length);
    System.out.println("[sift] loading sift_query.fvecs ...");
    float[][] queries = loadFvecs(dir.resolve("sift_query.fvecs"));
    System.out.println("[sift] loading sift_groundtruth.ivecs ...");
    int[][] gt = loadIvecs(dir.resolve("sift_groundtruth.ivecs"), 10);
    System.out.printf("[sift] queries=%d gt=%d x %d%n", queries.length, gt.length, gt[0].length);

    long t0 = System.currentTimeMillis();
    HnswGraph graph = HnswGraphBuilder.create(new Phase0Main.FloatArraySupplier(base), M, EF_C, 42)
        .build(base.length);
    long buildMs = System.currentTimeMillis() - t0;
    System.out.printf("[sift] HNSW built in %d s (levels=%d)%n", buildMs / 1000, graph.numLevels());

    int n = base.length;
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
    System.out.printf("[sift] upper-layer pool=%d (%.1f%%)%n", pool.size(), 100.0 * pool.size() / n);

    int[] random4 = new int[R];
    Random rndE = new Random(2026);
    Set<Integer> seen = new HashSet<>();
    int ei = 0;
    while (ei < R) {
      int cand = pool.get(rndE.nextInt(pool.size()));
      if (seen.add(cand)) {
        random4[ei++] = cand;
      }
    }

    Agg agg = new Agg();
    Random rnd = new Random(777);
    for (int qi = 0; qi < numQueries && qi < queries.length; qi++) {
      int qIdx = rnd.nextInt(queries.length);
      float[] q = queries[qIdx];
      InstrumentedSimSearch.ScorerFn scorer = ord -> -InstrumentedSearch.dist2(q, base[ord]);
      int[] ground = gt[qIdx];

      InstrumentedSimSearch.Result r1 = InstrumentedSimSearch.search(
          graph, n, scorer, K, 100, null);
      double baseRec = Phase0Real.recallAt(ground, r1.topK);
      double single1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
          graph, n, scorer, K, 1600, null).topK);

      List<float[]> unionBase = PhaseAFair.topKList(r1);
      double rRandom = PhaseAFair.evalEntries(graph, n, scorer, null, nodeLevel, random4,
          ground, unionBase);
      double rDir = PhaseAFair.evalDirAware(graph, n, scorer, null, nodeLevel, pool,
          q, base, ground, unionBase, rnd);
      double rFar = PhaseAFair.evalFarthest(graph, n, scorer, null, nodeLevel, pool,
          r1.descendedNode, base, ground, unionBase);
      double rMarked = PhaseAFair.evalMarked(graph, n, scorer, null, nodeLevel, pool,
          r1.descendedNode, base, ground, unionBase, r1, rnd);
      // probe-budget variants of the marked mechanism (SIFT follow-up)
      double m200 = markedWithProbe(graph, n, scorer, nodeLevel, pool, base, ground, 200, rnd);
      double m400 = markedWithProbe(graph, n, scorer, nodeLevel, pool, base, ground, 400, rnd);

      agg.add(baseRec, single1600, rRandom, rDir, rFar, rMarked, m200, m400);
    }

    agg.report(System.out, tag);
    Path outDir = Paths.get(System.getProperty("out.dir", "out"));
    Files.createDirectories(outDir);
    try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
        outDir.resolve("phaseSIFT-" + tag + ".csv"), StandardCharsets.UTF_8))) {
      pw.println("baseRec,single1600,random4,dirAware4,farthest4,marked4,markedProbe200,markedProbe400");
      for (String line : agg.csvLines) {
        pw.println(line);
      }
    }
    System.out.printf("[sift] wrote out/phaseSIFT-%s.csv%n", tag);
  }

  /** Marked mechanism with a given PROBE budget: search(probe) → mark → 4×400
   *  farthest-unvisited beams → union (probe's top-k included). */
  static double markedWithProbe(HnswGraph graph, int n, InstrumentedSimSearch.ScorerFn scorer,
                                int[] nodeLevel, List<Integer> pool, float[][] base,
                                int[] ground, int probeBudget, Random rnd) throws IOException {
    InstrumentedSimSearch.Result probe = InstrumentedSimSearch.search(
        graph, n, scorer, K, probeBudget, null);
    return PhaseAFair.evalMarked(graph, n, scorer, null, nodeLevel, pool,
        probe.descendedNode, base, ground, PhaseAFair.topKList(probe), probe, rnd);
  }

  // ---------- loaders (little-endian) ----------

  static float[][] loadFvecs(Path p) throws IOException {
    try (DataInputStream in = new DataInputStream(new BufferedInputStream(
        Files.newInputStream(p), 1 << 20))) {
      List<float[]> list = new ArrayList<>(1_000_000);
      while (in.available() > 0) {
        int dim = Integer.reverseBytes(in.readInt());
        float[] v = new float[dim];
        byte[] buf = new byte[dim * 4];
        in.readFully(buf);
        for (int j = 0; j < dim; j++) {
          int b0 = buf[4 * j] & 0xff;
          int b1 = (buf[4 * j + 1] & 0xff) << 8;
          int b2 = (buf[4 * j + 2] & 0xff) << 16;
          int b3 = (buf[4 * j + 3] & 0xff) << 24;
          v[j] = Float.intBitsToFloat(b0 | b1 | b2 | b3);
        }
        list.add(v);
      }
      return list.toArray(new float[0][]);
    }
  }

  static int[][] loadIvecs(Path p, int k) throws IOException {
    try (DataInputStream in = new DataInputStream(new BufferedInputStream(
        Files.newInputStream(p), 1 << 20))) {
      List<int[]> list = new ArrayList<>(10_000);
      while (in.available() > 0) {
        int kk = Integer.reverseBytes(in.readInt());
        int[] row = new int[Math.min(k, kk)];
        for (int j = 0; j < kk; j++) {
          int v = Integer.reverseBytes(in.readInt());
          if (j < k) {
            row[j] = v;
          }
        }
        list.add(row);
      }
      return list.toArray(new int[0][]);
    }
  }

  // ---------- aggregation (same shape as PhaseT2I) ----------

  static final class Agg {
    final List<String> csvLines = new ArrayList<>();
    int n = 0;
    double base = 0, s16 = 0, rnd = 0, dir = 0, far = 0, marked = 0, mk200 = 0, mk400 = 0;
    int bRnd = 0, bDir = 0, bFar = 0, bMarked = 0, bMk200 = 0, bMk400 = 0;
    int hn = 0;
    double hBase = 0, hS16 = 0, hRnd = 0, hDir = 0, hFar = 0, hMarked = 0;
    int hbRnd = 0, hbDir = 0, hbFar = 0, hbMarked = 0;

    void add(double b, double s, double a, double d, double f, double m, double m200,
             double m400) {
      n++;
      base += b;
      s16 += s;
      rnd += a;
      dir += d;
      far += f;
      marked += m;
      mk200 += m200;
      mk400 += m400;
      if (a > s + 0.01) bRnd++;
      if (d > s + 0.01) bDir++;
      if (f > s + 0.01) bFar++;
      if (m > s + 0.01) bMarked++;
      if (m200 > s + 0.01) bMk200++;
      if (m400 > s + 0.01) bMk400++;
      if (b < 0.9) {
        hn++;
        hBase += b;
        hS16 += s;
        hRnd += a;
        hDir += d;
        hFar += f;
        hMarked += m;
        if (a > s + 0.01) hbRnd++;
        if (d > s + 0.01) hbDir++;
        if (f > s + 0.01) hbFar++;
        if (m > s + 0.01) hbMarked++;
      }
      csvLines.add(String.format(Locale.ROOT, "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
          b, s, a, d, f, m, m200, m400));
    }

    void report(Appendable out, String tag) throws IOException {
      out.append(String.format(Locale.ROOT, "%n==== PhaseSIFT %s (all queries n=%d) ====%n",
          tag, n));
      out.append(String.format(Locale.ROOT,
          "  base@100=%.3f  single@1600=%.3f%n", base / n, s16 / n));
      row(out, "random4", rnd / n, bRnd, n);
      row(out, "dirAware4", dir / n, bDir, n);
      row(out, "farthest4", far / n, bFar, n);
      row(out, "marked4(probe100)", marked / n, bMarked, n);
      row(out, "marked4(probe200)", mk200 / n, bMk200, n);
      row(out, "marked4(probe400)", mk400 / n, bMk400, n);
      if (hn > 0) {
        out.append(String.format(Locale.ROOT,
            "%n  -- hard queries (base@100<0.9, n=%d) --%n", hn));
        out.append(String.format(Locale.ROOT,
            "  base@100=%.3f  single@1600=%.3f%n", hBase / hn, hS16 / hn));
        row(out, "random4", hRnd / hn, hbRnd, hn);
        row(out, "dirAware4", hDir / hn, hbDir, hn);
        row(out, "farthest4", hFar / hn, hbFar, hn);
        row(out, "marked4", hMarked / hn, hbMarked, hn);
      }
    }

    void row(Appendable out, String name, double recall, int breaks, int total)
        throws IOException {
      out.append(String.format(Locale.ROOT,
          "  %-11s: %.3f  (breaks %d/%d = %.1f%%)%n",
          name, recall, breaks, total, 100.0 * breaks / total));
    }
  }
}
