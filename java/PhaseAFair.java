package phase0;

import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
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
import java.util.Random;
import java.util.Set;

/**
 * FAIR comparison (user-requested): the user's entry-selection mechanisms vs
 * random entries, ALL with per-entry budget 400 + score union, on the SAME
 * queries and datasets (real medical dense_vec + heterogeneous synthetic).
 *
 * Variants (4 entries each, budget 400 per entry, union with base@100 top-k):
 *   random    : fixed random upper-layer entries (0 discovery cost)
 *   dirAware  : user's direction-aware jump — top-4 pool entries closest to the
 *               QUERY from a 100-sample (discovery cost 100 dc)
 *   farthest  : user's restart family — top-4 pool entries farthest from the
 *               descended node (discovery ~0, precomputable per query)
 *   marked    : user's path-marking idea — sequential restarts, each skipping
 *               all previously visited nodes (shared visited), budget 400/step
 *
 * <p>Usage: PhaseAFair &lt;indexDir&gt; &lt;field&gt; &lt;numQueries&gt; &lt;tag&gt;   (real data)
 *        PhaseAFair hetero                                      (synthetic)
 */
public final class PhaseAFair {

  static final int K = 10;
  static final int R = 4;
  static final int BUDGET = 400;

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length >= 1 && args[0].equals("hetero")) {
      runHetero();
      return;
    }
    if (args.length < 4) {
      System.err.println("usage: PhaseAFair <indexDir> <field> <numQueries> <tag> [seedOffset]");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int numQueries = Integer.parseInt(args[2]);
    String tag = args[3];
    long seedOffset = args.length >= 5 ? Long.parseLong(args[4]) : 0;
    runReal(indexDir, field, numQueries, tag, seedOffset);
  }

  // ---------------- real data ----------------

  static void runReal(Path indexDir, String field, int numQueries, String tag, long seedOffset)
      throws IOException {
    System.out.printf("[fair] real indexDir=%s field=%s queries=%d tag=%s seedOffset=%d%n",
        indexDir, field, numQueries, tag, seedOffset);
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
        int[] nodeLevel = levels(graph, size);
        List<Integer> pool = poolOf(graph, nodeLevel);
        if (pool.size() < R) {
          continue;
        }
        org.apache.lucene.util.Bits accept = acceptBits(ctx, fvv, field, size);

        int[] random4 = randomEntries(pool, R, new Random(2026L * ctx.docBase + size));
        int qPerLeaf = Math.max(1, numQueries / leaves.size());
        Random rnd = new Random(7777L * ctx.docBase + size + seedOffset);
        for (int q = 0; q < qPerLeaf; q++) {
          int qOrd = rnd.nextInt(size);
          float[] qNorm = Phase0Real.normalize(vecs[qOrd]);
          InstrumentedSimSearch.ScorerFn scorer = ord -> Phase0Real.dot(qNorm, vecs[ord]);
          InstrumentedSimSearch.Result r1 = InstrumentedSimSearch.search(
              graph, size, scorer, K, 100, accept);
          int[] ground = Phase0Real.bruteForceTopK(vecs, qNorm, K, accept);
          double baseRec = Phase0Real.recallAt(ground, r1.topK);
          if (baseRec >= 0.9) {
            continue;
          }
          double single1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
              graph, size, scorer, K, 1600, accept).topK);

          List<float[]> unionBase = topKList(r1);
          double rRandom = evalEntries(graph, size, scorer, accept, nodeLevel, random4,
              ground, unionBase);
          double rDir = evalDirAware(graph, size, scorer, accept, nodeLevel, pool, qNorm,
              vecs, ground, unionBase, rnd);
          double rFar = evalFarthest(graph, size, scorer, accept, nodeLevel, pool,
              r1.descendedNode, vecs, ground, unionBase);
          double rMarked = evalMarked(graph, size, scorer, accept, nodeLevel, pool,
              r1.descendedNode, vecs, ground, unionBase, r1, rnd);
          // probe-400 variant of the marked mechanism
          InstrumentedSimSearch.Result probe400 = InstrumentedSimSearch.search(
              graph, size, scorer, K, 400, accept);
          double rMarkedP400 = evalMarked(graph, size, scorer, accept, nodeLevel, pool,
              probe400.descendedNode, vecs, ground, topKList(probe400), probe400, rnd);

          agg.add(baseRec, single1600, rRandom, rDir, rFar, rMarked, rMarkedP400);
        }
      }
      agg.report(System.out, "real-" + tag);
      writeCsv(agg, "phaseA-fair-" + tag + ".csv");
    }
  }

  // ---------------- hetero ----------------

  static void runHetero() throws IOException {
    int n = 9600 + 5 * 480;
    float[][] vecs = new float[n][32];
    Random r = new Random(42);
    float[][] centers = new float[5][32];
    for (int c = 0; c < 5; c++) {
      double norm = 0;
      for (int d = 0; d < 32; d++) {
        centers[c][d] = (float) r.nextGaussian();
        norm += centers[c][d] * centers[c][d];
      }
      float s = (float) (8.0 / Math.sqrt(norm));
      for (int d = 0; d < 32; d++) {
        centers[c][d] *= s;
      }
    }
    int idx = 0;
    for (int i = 0; i < 9600; i++) {
      for (int d = 0; d < 32; d++) {
        vecs[idx][d] = (float) (r.nextGaussian() * 0.8);
      }
      idx++;
    }
    for (int c = 0; c < 5; c++) {
      for (int i = 0; i < 480; i++) {
        for (int d = 0; d < 32; d++) {
          vecs[idx][d] = centers[c][d] + (float) (r.nextGaussian() * 0.15);
        }
        idx++;
      }
    }
    List<Integer> qOrds = new ArrayList<>();
    List<Integer> qCluster = new ArrayList<>();
    Random qr = new Random(7);
    for (int i = 0; i < 150; i++) {
      qOrds.add(qr.nextInt(9600));
      qCluster.add(-1);
    }
    for (int c = 0; c < 5; c++) {
      for (int i = 0; i < 50; i++) {
        qOrds.add(9600 + c * 480 + qr.nextInt(480));
        qCluster.add(c);
      }
    }
    RandomVectorScorerSupplier sup = new Phase0Main.FloatArraySupplier(vecs);
    HnswGraph graph = HnswGraphBuilder.create(sup, 16, 100, 42).build(n);
    int[] nodeLevel = levels(graph, n);
    List<Integer> pool = poolOf(graph, nodeLevel);
    int[] random4 = randomEntries(pool, R, new Random(2026));

    Agg big = new Agg(), small = new Agg();
    for (int qi = 0; qi < qOrds.size(); qi++) {
      int qOrd = qOrds.get(qi);
      float[] q = vecs[qOrd];
      InstrumentedSimSearch.ScorerFn scorer = ord -> -InstrumentedSearch.dist2(q, vecs[ord]);
      int[] ground = bruteForce(vecs, q, K);
      InstrumentedSimSearch.Result r1 = InstrumentedSimSearch.search(
          graph, n, scorer, K, 100, null);
      double baseRec = Phase0Real.recallAt(ground, r1.topK);
      if (baseRec >= 0.9) {
        continue;
      }
      double single1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
          graph, n, scorer, K, 1600, null).topK);
      List<float[]> unionBase = topKList(r1);
      Random rnd = new Random(qi * 31L);
      double rRandom = evalEntries(graph, n, scorer, null, nodeLevel, random4, ground, unionBase);
      double rDir = evalDirAware(graph, n, scorer, null, nodeLevel, pool, q, vecs, ground,
          unionBase, rnd);
      double rFar = evalFarthest(graph, n, scorer, null, nodeLevel, pool, r1.descendedNode,
          vecs, ground, unionBase);
      double rMarked = evalMarked(graph, n, scorer, null, nodeLevel, pool, r1.descendedNode,
          vecs, ground, unionBase, r1, rnd);
      InstrumentedSimSearch.Result probe400 = InstrumentedSimSearch.search(
          graph, n, scorer, K, 400, null);
      double rMarkedP400 = evalMarked(graph, n, scorer, null, nodeLevel, pool,
          probe400.descendedNode, vecs, ground, topKList(probe400), probe400, rnd);
      Agg a = qCluster.get(qi) < 0 ? big : small;
      a.add(baseRec, single1600, rRandom, rDir, rFar, rMarked, rMarkedP400);
    }
    System.out.printf(Locale.ROOT, "%n==== fair hetero ====%n");
    big.report(System.out, "hetero-MEGA");
    small.report(System.out, "hetero-SMALL");
  }

  // ---------------- variant evaluators ----------------

  static double evalEntries(HnswGraph graph, int size, InstrumentedSimSearch.ScorerFn scorer,
                            org.apache.lucene.util.Bits accept, int[] nodeLevel, int[] entries,
                            int[] ground, List<float[]> unionBase) throws IOException {
    List<float[]> union = new ArrayList<>(unionBase);
    for (int e : entries) {
      if (e < 0 || nodeLevel[e] < 0) {
        continue;
      }
      InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
          graph, size, scorer, K, BUDGET, accept, e, nodeLevel[e]);
      for (int j = 0; j < rr.topK.length; j++) {
        union.add(new float[]{rr.topK[j], rr.topKScores[j]});
      }
    }
    return unionRecall(ground, union, K);
  }

  static double evalDirAware(HnswGraph graph, int size, InstrumentedSimSearch.ScorerFn scorer,
                             org.apache.lucene.util.Bits accept, int[] nodeLevel, List<Integer> pool,
                             float[] qNorm, float[][] vecs, int[] ground,
                             List<float[]> unionBase, Random rnd) throws IOException {
    // discovery: sample 100 pool entries, keep 4 closest to the query
    float[] bestD = {Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE};
    int[] bestN = {-1, -1, -1, -1};
    Set<Integer> seen = new HashSet<>();
    for (int i = 0; i < 100; i++) {
      int cand = pool.get(rnd.nextInt(pool.size()));
      if (!seen.add(cand)) {
        continue;
      }
      float d = InstrumentedSearch.dist2(qNorm, vecs[cand]);
      for (int j = 0; j < R; j++) {
        if (d < bestD[j]) {
          for (int t = R - 1; t > j; t--) {
            bestD[t] = bestD[t - 1];
            bestN[t] = bestN[t - 1];
          }
          bestD[j] = d;
          bestN[j] = cand;
          break;
        }
      }
    }
    return evalEntries(graph, size, scorer, accept, nodeLevel, bestN, ground, unionBase);
  }

  static double evalFarthest(HnswGraph graph, int size, InstrumentedSimSearch.ScorerFn scorer,
                             org.apache.lucene.util.Bits accept, int[] nodeLevel, List<Integer> pool,
                             int from, float[][] vecs, int[] ground, List<float[]> unionBase)
      throws IOException {
    float[] bestD = {-1, -1, -1, -1};
    int[] bestN = {-1, -1, -1, -1};
    for (int cand : pool) {
      float d = InstrumentedSearch.dist2(vecs[cand], vecs[from]);
      for (int j = 0; j < R; j++) {
        if (d > bestD[j]) {
          for (int t = R - 1; t > j; t--) {
            bestD[t] = bestD[t - 1];
            bestN[t] = bestN[t - 1];
          }
          bestD[j] = d;
          bestN[j] = cand;
          break;
        }
      }
    }
    return evalEntries(graph, size, scorer, accept, nodeLevel, bestN, ground, unionBase);
  }

  static double evalMarked(HnswGraph graph, int size, InstrumentedSimSearch.ScorerFn scorer,
                           org.apache.lucene.util.Bits accept, int[] nodeLevel, List<Integer> pool,
                           int from, float[][] vecs, int[] ground, List<float[]> unionBase,
                           InstrumentedSimSearch.Result base, Random rnd) throws IOException {
    List<float[]> union = new ArrayList<>(unionBase);
    java.util.BitSet marked = new java.util.BitSet(size);
    marked.or(base.visited);
    for (int step = 0; step < R; step++) {
      // farthest UNVISITED pool entry from the descended node
      int best = -1;
      float bestD = -1;
      int attempts = 0, found = 0;
      while (found < 500 && attempts < 4000) {
        attempts++;
        int cand = pool.get(rnd.nextInt(pool.size()));
        if (marked.get(cand)) {
          continue;
        }
        found++;
        float d = InstrumentedSearch.dist2(vecs[cand], vecs[from]);
        if (d > bestD) {
          bestD = d;
          best = cand;
        }
      }
      if (best < 0) {
        break;
      }
      InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
          graph, size, scorer, K, BUDGET, accept, best, nodeLevel[best], marked);
      marked.or(rr.visited);
      for (int j = 0; j < rr.topK.length; j++) {
        union.add(new float[]{rr.topK[j], rr.topKScores[j]});
      }
    }
    return unionRecall(ground, union, K);
  }

  // ---------------- shared helpers ----------------

  static int[] levels(HnswGraph graph, int size) throws IOException {
    int[] lvl = new int[size];
    java.util.Arrays.fill(lvl, -1);
    for (int level = 1; level <= graph.numLevels() - 1; level++) {
      var it = graph.getNodesOnLevel(level);
      while (it.hasNext()) {
        lvl[it.nextInt()] = level;
      }
    }
    return lvl;
  }

  static List<Integer> poolOf(HnswGraph graph, int[] lvl) {
    List<Integer> pool = new ArrayList<>();
    for (int i = 0; i < lvl.length; i++) {
      if (lvl[i] >= 1) {
        pool.add(i);
      }
    }
    return pool;
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

  static org.apache.lucene.util.Bits acceptBits(LeafReaderContext ctx, FloatVectorValues fvv,
                                                String field, int size) throws IOException {
    org.apache.lucene.util.Bits liveDocs = ctx.reader().getLiveDocs();
    RandomAccessVectorValues rav = fvv instanceof RandomAccessVectorValues r ? r : null;
    final RandomAccessVectorValues ravF = rav;
    return (liveDocs != null && ravF != null) ? new org.apache.lucene.util.Bits() {
      @Override
      public boolean get(int ord) {
        return liveDocs.get(ravF.ordToDoc(ord));
      }

      @Override
      public int length() {
        return size;
      }
    } : null;
  }

  static List<float[]> topKList(InstrumentedSimSearch.Result r) {
    List<float[]> out = new ArrayList<>();
    for (int i = 0; i < r.topK.length; i++) {
      out.add(new float[]{r.topK[i], r.topKScores[i]});
    }
    return out;
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

  static void writeCsv(Agg agg, String name) throws IOException {
    Path outDir = Paths.get(System.getProperty("out.dir", "out"));
    Files.createDirectories(outDir);
    try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
        outDir.resolve(name), StandardCharsets.UTF_8))) {
      pw.println("baseRec,single1600,random4,dirAware4,farthest4,marked4,markedP400");
      for (String line : agg.csvLines) {
        pw.println(line);
      }
    }
  }

  static final class Agg {
    final List<String> csvLines = new ArrayList<>();
    double base = 0, s16 = 0, rnd = 0, dir = 0, far = 0, marked = 0, mk400 = 0;
    int n = 0, bRnd = 0, bDir = 0, bFar = 0, bMarked = 0, bMk400 = 0;

    void add(double b, double s, double a, double d, double f, double m, double mp400) {
      n++;
      base += b;
      s16 += s;
      rnd += a;
      dir += d;
      far += f;
      marked += m;
      mk400 += mp400;
      if (a > s + 0.01) bRnd++;
      if (d > s + 0.01) bDir++;
      if (f > s + 0.01) bFar++;
      if (m > s + 0.01) bMarked++;
      if (mp400 > s + 0.01) bMk400++;
      csvLines.add(String.format(Locale.ROOT, "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
          b, s, a, d, f, m, mp400));
    }

    void report(Appendable out, String label) throws IOException {
      out.append(String.format(Locale.ROOT,
          "%n[%s n=%d]  base@100=%.3f  single@1600=%.3f%n", label, n, base / n, s16 / n));
      out.append(String.format(Locale.ROOT,
          "  random4    : %.3f (breaks %d/%d = %.1f%%)%n", rnd / n, bRnd, n, 100.0 * bRnd / n));
      out.append(String.format(Locale.ROOT,
          "  dirAware4  : %.3f (breaks %d/%d = %.1f%%)%n", dir / n, bDir, n, 100.0 * bDir / n));
      out.append(String.format(Locale.ROOT,
          "  farthest4  : %.3f (breaks %d/%d = %.1f%%)%n", far / n, bFar, n, 100.0 * bFar / n));
      out.append(String.format(Locale.ROOT,
          "  marked4    : %.3f (breaks %d/%d = %.1f%%)%n", marked / n, bMarked, n, 100.0 * bMarked / n));
      out.append(String.format(Locale.ROOT,
          "  marked4(p400): %.3f (breaks %d/%d = %.1f%%)%n", mk400 / n, bMk400, n, 100.0 * bMk400 / n));
    }
  }
}
