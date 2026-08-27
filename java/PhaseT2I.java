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
 * Real heterogeneous validation: Yandex Text-to-Image 1M (cross-modal OOD).
 * Loads fbin base/queries + ibin ground truth, builds a real HNSW with
 * Lucene's builder (M=16, efC=100), and runs the FAIR four-strategy
 * comparison (marked sequential / random / direction-aware / farthest),
 * all with per-entry budget 400 + score union, vs single@400/@1600.
 *
 * <p>Usage: PhaseT2I &lt;dataDir&gt; &lt;numQueries&gt; &lt;tag&gt;
 */
public final class PhaseT2I {

  static final int K = 10;
  static final int M = 16;
  static final int EF_C = 100;
  static final int R = 4;
  static final int BUDGET = 400;

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 3) {
      System.err.println("usage: PhaseT2I <dataDir> <numQueries> <tag>");
      System.exit(2);
    }
    Path dir = Paths.get(args[0]);
    int numQueries = Integer.parseInt(args[1]);
    String tag = args[2];

    System.out.printf("[t2i] dataDir=%s queries=%d tag=%s%n", dir, numQueries, tag);

    System.out.println("[t2i] loading base.1M.fbin ...");
    float[][] base = loadFbin(dir.resolve("base.1M.fbin"));
    System.out.printf("[t2i] base: %d x %d%n", base.length, base[0].length);
    System.out.println("[t2i] loading query.public.100K.fbin ...");
    float[][] queries = loadFbin(dir.resolve("query.public.100K.fbin"));
    System.out.println("[t2i] loading groundtruth.public.100K.ibin ...");
    int[][] gt = loadIbin(dir.resolve("groundtruth.public.100K.ibin"), 10);
    System.out.printf("[t2i] queries=%d gt=%d x %d%n", queries.length, gt.length, gt[0].length);

    // sanity checks: GT index range, vector norms, self-similarity
    int gtMax = 0, gtMin = Integer.MAX_VALUE;
    for (int[] row : gt) {
      for (int idx : row) {
        gtMax = Math.max(gtMax, idx);
        gtMin = Math.min(gtMin, idx);
      }
    }
    float norm0 = 0f;
    for (float x : base[0]) {
      norm0 += x * x;
    }
    System.out.printf("[t2i] sanity: gt range [%d,%d] (base size %d), norm(base[0])=%.4f, "
            + "selfcos(base[0])=%.4f%n",
        gtMin, gtMax, base.length, Math.sqrt(norm0),
        Phase0Real.dot(Phase0Real.normalize(base[0]), Phase0Real.normalize(base[0])));
    if (gtMax >= base.length || gtMin < 0) {
      throw new IllegalStateException("GT indices out of range — parsing wrong");
    }

    // ---- build real HNSW ----
    long t0 = System.currentTimeMillis();
    HnswGraph graph = HnswGraphBuilder.create(new Phase1Build.CosineSupplier(base), M, EF_C, 42)
        .build(base.length);
    long buildMs = System.currentTimeMillis() - t0;
    System.out.printf("[t2i] HNSW built in %d s (levels=%d)%n", buildMs / 1000, graph.numLevels());

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
    System.out.printf("[t2i] upper-layer pool=%d (%.1f%%)%n", pool.size(), 100.0 * pool.size() / n);

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

    // ---- evaluate (two candidate metrics; GT was computed with one of them) ----
    Agg aggCos = new Agg(), aggIp = new Agg();
    Random rnd = new Random(777);
    int evaluated = 0;
    for (int qi = 0; qi < numQueries && qi < queries.length; qi++) {
      int qIdx = rnd.nextInt(queries.length);
      int[] ground = gt[qIdx];
      float[] qNorm = Phase0Real.normalize(queries[qIdx]);

      InstrumentedSimSearch.ScorerFn scorerCos = ord -> Phase0Real.dot(qNorm, base[ord]);
      double s1600Cos = evalOne(graph, n, scorerCos, qNorm, base, ground, nodeLevel, pool,
          random4, rnd, aggCos);
      // probe-budget variants of the marked mechanism (cos metric; the two
      // metrics agree to <=0.003, so cos suffices for this sweep)
      double mk200 = markedWithProbe(graph, n, scorerCos, nodeLevel, pool, base, ground, 200, rnd);
      double mk400 = markedWithProbe(graph, n, scorerCos, nodeLevel, pool, base, ground, 400, rnd);
      aggCos.addMarkedProbe(mk200, mk400, s1600Cos);

      float[] qRaw = queries[qIdx];
      InstrumentedSimSearch.ScorerFn scorerIp = ord -> Phase0Real.dot(qRaw, base[ord]);
      evalOne(graph, n, scorerIp, qNorm, base, ground, nodeLevel, pool, random4, rnd, aggIp);

      evaluated++;
      if (evaluated % 200 == 0) {
        System.out.printf("[t2i] evaluated %d queries%n", evaluated);
      }
    }

    aggCos.report(System.out, tag + "-cos");
    aggIp.report(System.out, tag + "-ip");
    Path outDir = Paths.get(System.getProperty("out.dir", "out"));
    Files.createDirectories(outDir);
    writeCsv(aggCos, outDir.resolve("phaseT2I-" + tag + "-cos.csv"));
    writeCsv(aggIp, outDir.resolve("phaseT2I-" + tag + "-ip.csv"));
    System.out.printf("[t2i] wrote CSVs%n", tag);
  }

  static double evalOne(HnswGraph graph, int n, InstrumentedSimSearch.ScorerFn scorer,
                       float[] qForSelection, float[][] base, int[] ground, int[] nodeLevel,
                       List<Integer> pool, int[] random4, Random rnd, Agg agg) throws IOException {
    InstrumentedSimSearch.Result r1 = InstrumentedSimSearch.search(
        graph, n, scorer, K, 100, null);
    double baseRec = Phase0Real.recallAt(ground, r1.topK);
    double single1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
        graph, n, scorer, K, 1600, null).topK);

    List<float[]> unionBase = PhaseAFair.topKList(r1);
    double rRandom = PhaseAFair.evalEntries(graph, n, scorer, null, nodeLevel, random4,
        ground, unionBase);
    double rDir = PhaseAFair.evalDirAware(graph, n, scorer, null, nodeLevel, pool,
        qForSelection, base, ground, unionBase, rnd);
    double rFar = PhaseAFair.evalFarthest(graph, n, scorer, null, nodeLevel, pool,
        r1.descendedNode, base, ground, unionBase);
    double rMarked = PhaseAFair.evalMarked(graph, n, scorer, null, nodeLevel, pool,
        r1.descendedNode, base, ground, unionBase, r1, rnd);

    agg.add(baseRec, single1600, rRandom, rDir, rFar, rMarked);
    return single1600;
  }

  /** Marked mechanism with a given PROBE budget (probe → mark → 4×400). */
  static double markedWithProbe(HnswGraph graph, int n, InstrumentedSimSearch.ScorerFn scorer,
                                int[] nodeLevel, List<Integer> pool, float[][] base,
                                int[] ground, int probeBudget, Random rnd) throws IOException {
    InstrumentedSimSearch.Result probe = InstrumentedSimSearch.search(
        graph, n, scorer, K, probeBudget, null);
    return PhaseAFair.evalMarked(graph, n, scorer, null, nodeLevel, pool,
        probe.descendedNode, base, ground, PhaseAFair.topKList(probe), probe, rnd);
  }

  static void writeCsv(Agg agg, Path out) throws IOException {
    try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
      pw.println("baseRec,single1600,random4,dirAware4,farthest4,marked4");
      for (String line : agg.csvLines) {
        pw.println(line);
      }
    }
  }

  // ---------- loaders ----------

  static float[][] loadFbin(Path p) throws IOException {
    try (DataInputStream in = new DataInputStream(new BufferedInputStream(
        Files.newInputStream(p), 1 << 20))) {
      int n = Integer.reverseBytes(in.readInt());
      int dim = Integer.reverseBytes(in.readInt());
      float[][] v = new float[n][dim];
      byte[] buf = new byte[dim * 4];
      for (int i = 0; i < n; i++) {
        in.readFully(buf);
        for (int j = 0; j < dim; j++) {
          // LITTLE-endian float32: buf[0] is the least significant byte
          int b0 = buf[4 * j] & 0xff;
          int b1 = (buf[4 * j + 1] & 0xff) << 8;
          int b2 = (buf[4 * j + 2] & 0xff) << 16;
          int b3 = (buf[4 * j + 3] & 0xff) << 24;
          v[i][j] = Float.intBitsToFloat(b0 | b1 | b2 | b3);
        }
      }
      return v;
    }
  }

  static int[][] loadIbin(Path p, int k) throws IOException {
    try (DataInputStream in = new DataInputStream(new BufferedInputStream(
        Files.newInputStream(p), 1 << 20))) {
      int n = Integer.reverseBytes(in.readInt());
      int kk = Integer.reverseBytes(in.readInt());
      if (kk != k) {
        throw new IllegalStateException("k mismatch: " + kk);
      }
      int[][] gt = new int[n][k];
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
          gt[i][j] = Integer.reverseBytes(in.readInt());
        }
      }
      return gt;
    }
  }

  // ---------- aggregation ----------

  static final class Agg {
    final List<String> csvLines = new ArrayList<>();
    int n = 0;
    double base = 0, s16 = 0, rnd = 0, dir = 0, far = 0, marked = 0;
    double mk200 = 0, mk400 = 0;
    int bRnd = 0, bDir = 0, bFar = 0, bMarked = 0, bMk200 = 0, bMk400 = 0;
    // hard subset (base < 0.9)
    int hn = 0;
    double hBase = 0, hS16 = 0, hRnd = 0, hDir = 0, hFar = 0, hMarked = 0;
    int hbRnd = 0, hbDir = 0, hbFar = 0, hbMarked = 0;

    void add(double b, double s, double a, double d, double f, double m) {
      n++;
      base += b;
      s16 += s;
      rnd += a;
      dir += d;
      far += f;
      marked += m;
      if (a > s + 0.01) bRnd++;
      if (d > s + 0.01) bDir++;
      if (f > s + 0.01) bFar++;
      if (m > s + 0.01) bMarked++;
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
      csvLines.add(String.format(Locale.ROOT, "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
          b, s, a, d, f, m));
    }

    /** Attach probe-sweep variants after the main add (cos metric only). */
    void addMarkedProbe(double m200, double m400, double single1600) {
      mk200 += m200;
      mk400 += m400;
      if (m200 > single1600 + 0.01) bMk200++;
      if (m400 > single1600 + 0.01) bMk400++;
    }

    void report(Appendable out, String tag) throws IOException {
      out.append(String.format(Locale.ROOT, "%n==== PhaseT2I %s (all queries n=%d) ====%n",
          tag, n));
      out.append(String.format(Locale.ROOT,
          "  base@100=%.3f  single@1600=%.3f%n", base / n, s16 / n));
      row(out, "random4", rnd / n, bRnd, n);
      row(out, "dirAware4", dir / n, bDir, n);
      row(out, "farthest4", far / n, bFar, n);
      row(out, "marked4(p100)", marked / n, bMarked, n);
      row(out, "marked4(p200)", mk200 / n, bMk200, n);
      row(out, "marked4(p400)", mk400 / n, bMk400, n);
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
