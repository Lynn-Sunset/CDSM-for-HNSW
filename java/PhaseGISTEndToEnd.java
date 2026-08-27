package phase0;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * GIST1M real-Lucene END-TO-END sweep, production segment shape:
 *   - real mmap Lucene 9.12 index (PhaseGISTIndex, forceMerged to 1 segment)
 *   - real HnswGraph per leaf via HnswGraphProvider.getGraph
 *   - REAL DefaultFlatVectorScorer (EUCLIDEAN, Panama SIMD)
 *   - PER-LEAF search + CROSS-LEAF reduce by score (ES per-segment knn semantics),
 *     ord -> doc mapping via FloatVectorValues.ordToDoc
 *   - official GIST GT (global doc ids)
 * Configs: gated single {100..12800}, ungated single {1600,6400,12800},
 * mkL-R16x400, mkU-R4x1600, mkBU-R4x1600 (CDSM family per leaf).
 *
 * <p>Usage: PhaseGISTEndToEnd &lt;indexDir&gt; &lt;gistDir&gt; &lt;numQueries&gt;
 */
public final class PhaseGISTEndToEnd {

  static final int K = 10;
  static final int PROBE = 400;

  static final class Cfg {
    final String name;
    final int kind;      // 0 gated single | 4 ungated single | 1 mkL | 3 mkU | 7 mkBU
    final int budget;
    final int rounds;
    final int perBudget;

    Cfg(String name, int kind, int budget, int rounds, int perBudget) {
      this.name = name; this.kind = kind; this.budget = budget;
      this.rounds = rounds; this.perBudget = perBudget;
    }
  }

  static final Cfg[] CFGS = {
      new Cfg("single-100", 0, 100, 0, 0),
      new Cfg("single-400", 0, 400, 0, 0),
      new Cfg("single-1600", 0, 1600, 0, 0),
      new Cfg("single-6400", 0, 6400, 0, 0),
      new Cfg("single-12800", 0, 12800, 0, 0),
      new Cfg("singleU-1600", 4, 1600, 0, 0),
      new Cfg("singleU-6400", 4, 6400, 0, 0),
      new Cfg("singleU-12800", 4, 12800, 0, 0),
      new Cfg("mkL-R16x400", 1, 0, 16, 400),
      new Cfg("mkU-R4x1600", 3, 0, 4, 1600),
      new Cfg("mkBU-R4x1600", 7, 0, 4, 1600),
  };

  static final class Leaf {
    int docBase;
    int size;
    HnswGraph graph;
    FloatVectorValues fvv;
    RandomAccessVectorValues rav;
    int[] nodeLevel;
    int[] entryTable;     // upper-layer table
    int[] baseTable;      // all-node table

    int toDoc(int ord) throws IOException {
      return rav.ordToDoc(ord);
    }
  }

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 3) {
      System.err.println("usage: PhaseGISTEndToEnd <indexDir> <gistDir> <numQueries>");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    Path gistDir = Paths.get(args[1]);
    int numQueries = Integer.parseInt(args[2]);

    float[][] queries = loadFvecs(gistDir.resolve("gist_query.fvecs"), numQueries);
    System.out.printf("[gist-e2e] queries=%d%n", queries.length);

    try (DirectoryReader ir = DirectoryReader.open(FSDirectory.open(indexDir, NoLockFactory.INSTANCE))) {
      List<Leaf> leaves = new ArrayList<>();
      for (LeafReaderContext ctx : ir.leaves()) {
        CodecReader cr = (CodecReader) ctx.reader();
        FloatVectorValues fvv = cr.getFloatVectorValues("vec");
        if (fvv == null) {
          continue;
        }
        Leaf lf = new Leaf();
        lf.docBase = ctx.docBase;
        lf.size = fvv.size();
        lf.fvv = fvv;
        lf.graph = Phase0Real.getRealGraph((FSDirectory) ir.directory(), cr, "vec");
        lf.rav = fvv instanceof RandomAccessVectorValues r
            ? r
            : RandomAccessVectorValues.fromFloats(
                List.of(Phase0Real.loadVectors(fvv, lf.size, fvv.dimension())), fvv.dimension());
        lf.nodeLevel = new int[lf.size];
        Arrays.fill(lf.nodeLevel, -1);
        List<Integer> pool = new ArrayList<>();
        for (int level = 1; level <= lf.graph.numLevels() - 1; level++) {
          var it = lf.graph.getNodesOnLevel(level);
          while (it.hasNext()) {
            int n = it.nextInt();
            lf.nodeLevel[n] = level;
            pool.add(n);
          }
        }
        lf.entryTable = new int[16];
        Random rndE = new Random(2026L * ctx.docBase + lf.size);
        Set<Integer> seen = new HashSet<>();
        int ei = 0;
        while (ei < lf.entryTable.length && !pool.isEmpty()) {
          int cand = pool.get(rndE.nextInt(pool.size()));
          if (seen.add(cand)) lf.entryTable[ei++] = cand;
        }
        lf.baseTable = new int[16];
        Random rndB = new Random(2027L * ctx.docBase + lf.size);
        Set<Integer> seenB = new HashSet<>();
        int bi = 0;
        while (bi < lf.baseTable.length) {
          int cand = rndB.nextInt(lf.size);
          if (seenB.add(cand)) lf.baseTable[bi++] = cand;
        }
        leaves.add(lf);
      }
      System.out.printf("[gist-e2e] %d leaf/leaves (docBase,size):", leaves.size());
      for (Leaf lf : leaves) {
        System.out.printf(" (%d,%d)", lf.docBase, lf.size);
      }
      System.out.println();

      // Engine-space ground truth: exact top-10 over the index's OWN doc space
      // (Lucene merges renumber doc ids, so file-order GT cannot be joined by
      // position; the production analog is joining on _id. We compute brute
      // force with the REAL production VectorScorer and cache it.)
      int[][] gt = engineGroundTruth(leaves, queries, gistDir.resolve("gist-gt-engine.ivecs"));

      // warmup on query 0
      for (Cfg cfg : CFGS) {
        run(cfg, leaves, queries[0], null);
      }
      System.out.println("[gist-e2e] warmup done");

      int nCfg = CFGS.length;
      double[][] rec = new double[nCfg][numQueries];
      long[][] dc = new long[nCfg][numQueries];
      long[][] ns = new long[nCfg][numQueries];
      final int TIMING_ROUNDS = 3;   // shuffled-order medians (removes JIT-order bias)

      for (int q = 0; q < numQueries; q++) {
        long[][][] nss = new long[nCfg][TIMING_ROUNDS][1];
        for (int round = 0; round < TIMING_ROUNDS; round++) {
          List<Integer> order = new ArrayList<>();
          for (int c = 0; c < nCfg; c++) {
            order.add(c);
          }
          java.util.Collections.shuffle(order, new Random(4242L * (q + 1) + round));
          for (int c : order) {
            long t0 = System.nanoTime();
            RunOut ro = run(CFGS[c], leaves, queries[q], gt[q]);
            long t1 = System.nanoTime();
            nss[c][round][0] = t1 - t0;
            if (round == 0) {
              rec[c][q] = ro.recall;
              dc[c][q] = ro.dc;
            }
          }
        }
        for (int c = 0; c < nCfg; c++) {
          long[] ts = {nss[c][0][0], nss[c][1][0], nss[c][2][0]};
          Arrays.sort(ts);
          ns[c][q] = ts[TIMING_ROUNDS / 2];
        }
        if ((q + 1) % 250 == 0) {
          System.out.printf("[gist-e2e] %d/%d queries%n", q + 1, numQueries);
        }
      }

      System.out.printf(Locale.ROOT, "%n==== GIST real-Lucene end-to-end (n=%d, leaves=%d) ====%n",
          numQueries, leaves.size());
      System.out.printf(Locale.ROOT, "%-16s %8s %8s %8s %8s%n",
          "config", "recall", "dc", "p50ms", "us/dc");
      Path outDir = Paths.get(System.getProperty("out.dir", "out"));
      Files.createDirectories(outDir);
      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
          outDir.resolve("phaseGIST-endtoend.csv"), StandardCharsets.UTF_8))) {
        StringBuilder hdr = new StringBuilder("q");
        for (Cfg cfg : CFGS) {
          hdr.append(',').append(cfg.name).append("-rec");
        }
        for (Cfg cfg : CFGS) {
          hdr.append(',').append(cfg.name).append("-ns");
        }
        pw.println(hdr);
        for (int q = 0; q < numQueries; q++) {
          StringBuilder sb = new StringBuilder().append(q);
          for (int c = 0; c < nCfg; c++) {
            sb.append(',').append(String.format(Locale.ROOT, "%.4f", rec[c][q]));
          }
          for (int c = 0; c < nCfg; c++) {
            sb.append(',').append(ns[c][q]);
          }
          pw.println(sb);
        }
        for (int c = 0; c < nCfg; c++) {
          double r = Arrays.stream(rec[c]).average().orElse(0);
          double d = Arrays.stream(dc[c]).average().orElse(0);
          long[] t = ns[c].clone();
          Arrays.sort(t);
          double p50 = t[t.length / 2] / 1e6;
          System.out.printf(Locale.ROOT, "%-16s %.4f %8.0f %8.2f %8.3f%n",
              CFGS[c].name, r, d, p50, p50 * 1000.0 / d);
        }
      }
      System.out.println("[gist-e2e] wrote out/phaseGIST-endtoend.csv");
    }
  }

  /** Run one config across all leaves (sequential), merge by score, recall vs global GT. */
  static final class RunOut {
    double recall;
    long dc;
  }

  static RunOut run(Cfg cfg, List<Leaf> leaves, float[] query, int[] ground) throws IOException {
    List<long[]> global = new ArrayList<>();   // (docId, floatBits(score)) sorted desc
    long dcSum = 0;
    for (Leaf lf : leaves) {
      RandomVectorScorer rs = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
          VectorSimilarityFunction.EUCLIDEAN, lf.rav, query);
      InstrumentedSimSearch.ScorerFn scorer = ord -> {
        try {
          return rs.score(ord);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      };

      List<float[]> leafTop;   // {ord, score}
      if (cfg.kind == 0) {
        InstrumentedSimSearch.Result r = InstrumentedSimSearch.search(
            lf.graph, lf.size, scorer, K, cfg.budget, null);
        dcSum += r.scoreComps;
        leafTop = pack(r);
      } else if (cfg.kind == 4) {
        InstrumentedSimSearch.Result r = InstrumentedSimSearch.searchUngated(
            lf.graph, lf.size, scorer, K, cfg.budget, null);
        dcSum += r.scoreComps;
        leafTop = pack(r);
      } else {
        InstrumentedSimSearch.Result probe = InstrumentedSimSearch.search(
            lf.graph, lf.size, scorer, K, PROBE, null);
        dcSum += probe.scoreComps;
        List<float[]> union = new ArrayList<>(pack(probe));
        BitSet marked = new BitSet(lf.size);
        marked.or(probe.visited);
        int[] table = (cfg.kind == 7) ? lf.baseTable : lf.entryTable;
        for (int step = 0; step < cfg.rounds; step++) {
          int best = -1;
          for (int i = 0; i < table.length; i++) {
            int cand = table[(step + i) % table.length];
            if (!marked.get(cand)) {
              best = cand;
              break;
            }
          }
          if (best < 0) {
            break;
          }
          InstrumentedSimSearch.Result rr = (cfg.kind == 3 || cfg.kind == 7)
              ? InstrumentedSimSearch.searchUngated(
                  lf.graph, lf.size, scorer, K, cfg.perBudget, null, best,
                  lf.nodeLevel[best], marked)
              : InstrumentedSimSearch.searchFrom(
                  lf.graph, lf.size, scorer, K, cfg.perBudget, null, best,
                  lf.nodeLevel[best], marked);
          dcSum += rr.scoreComps;
          marked.or(rr.visited);
          marked.set(best);      // mark the entry node itself (descent->beam clear)
                                 // so deep rounds use distinct entries, no reuse
          union.addAll(pack(rr));
        }
        leafTop = union;
      }
      for (float[] e : leafTop) {
        global.add(new long[]{lf.toDoc((int) e[0]), Float.floatToIntBits(e[1])});
      }
    }
    global.sort((a, b) -> Float.compare(
        Float.intBitsToFloat((int) b[1]), Float.intBitsToFloat((int) a[1])));
    Set<Long> seen = new HashSet<>();
    int[] top = new int[K];
    int idx = 0;
    for (long[] e : global) {
      if (seen.add(e[0])) {
        top[idx++] = (int) e[0];
        if (idx == K) {
          break;
        }
      }
    }
    RunOut out = new RunOut();
    out.recall = ground == null ? 0.0 : Phase0Real.recallAt(ground, top);
    out.dc = dcSum;
    return out;
  }

  static List<float[]> pack(InstrumentedSimSearch.Result r) {
    List<float[]> out = new ArrayList<>(r.topK.length);
    for (int j = 0; j < r.topK.length; j++) {
      out.add(new float[]{r.topK[j], r.topKScores[j]});
    }
    return out;
  }

  /** Exact top-10 in engine doc space via the real production VectorScorer. */
  static int[][] engineGroundTruth(List<Leaf> leaves, float[][] queries, Path cacheFile)
      throws IOException {
    // Cache validation: the cache is only trusted when its sidecar signature matches
    // the current index shape (nDocs/dims/K), query count, and base-vector file
    // (size + head/tail SHA-256). A stale cache from a different index/data would
    // otherwise be loaded silently (external-review item 3: cache signature).
    Path sigFile = cacheFile.resolveSibling(cacheFile.getFileName() + ".sig");
    String wantSig = gtSignature(leaves, queries,
        cacheFile.getParent().resolve("gist_base.fvecs"));
    if (Files.exists(cacheFile) && Files.exists(sigFile)
        && wantSig.equals(Files.readString(sigFile).trim())) {
      ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(cacheFile)).order(ByteOrder.BIG_ENDIAN);
      int nq = buf.getInt();
      int[][] gt = new int[nq][];
      for (int q = 0; q < nq; q++) {
        gt[q] = new int[Math.min(K, buf.getInt())];
        for (int i = 0; i < gt[q].length; i++) {
          gt[q][i] = buf.getInt();
        }
      }
      System.out.printf("[gist-e2e] engine-space GT loaded from cache (%d queries, signature OK)%n", nq);
      return gt;
    }
    if (Files.exists(cacheFile)) {
      System.out.println("[gist-e2e] WARNING: GT cache signature mismatch (stale index/data?) — recomputing");
    }
    System.out.println("[gist-e2e] computing engine-space GT (brute force, ~25 min) ...");
    int nq = queries.length;
    int[][] gt = new int[nq][];
    long t0 = System.currentTimeMillis();
    for (int q = 0; q < nq; q++) {
      float[] bestS = new float[K + 1];
      int[] bestD = new int[K + 1];
      int cnt = 0;
      for (Leaf lf : leaves) {
        org.apache.lucene.search.VectorScorer vs = lf.fvv.scorer(queries[q]);
        var it = vs.iterator();
        for (int doc = it.nextDoc(); doc != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
             doc = it.nextDoc()) {
          float s = vs.score();
          bestD[cnt] = doc;
          bestS[cnt] = s;
          cnt++;
          for (int i = cnt - 1; i > 0 && bestS[i] > bestS[i - 1]; i--) {
            float ts = bestS[i]; bestS[i] = bestS[i - 1]; bestS[i - 1] = ts;
            int td = bestD[i]; bestD[i] = bestD[i - 1]; bestD[i - 1] = td;
          }
          if (cnt > K) {
            cnt = K;
          }
        }
      }
      gt[q] = java.util.Arrays.copyOf(bestD, cnt);
      if ((q + 1) % 100 == 0) {
        System.out.printf("[gist-e2e] GT %d/%d (%.0fs)%n", q + 1, nq,
            (System.currentTimeMillis() - t0) / 1000.0);
      }
    }
    try (java.io.DataOutputStream out = new java.io.DataOutputStream(
        Files.newOutputStream(cacheFile))) {
      out.writeInt(nq);
      for (int q = 0; q < nq; q++) {
        out.writeInt(gt[q].length);
        for (int i = 0; i < gt[q].length; i++) {
          out.writeInt(gt[q][i]);
        }
      }
    }
    Files.writeString(sigFile, wantSig);
    System.out.printf("[gist-e2e] engine-space GT cached to %s (signature written)%n", cacheFile);
    return gt;
  }

  /** Signature binding a GT cache to the index shape and base-vector data. */
  static String gtSignature(List<Leaf> leaves, float[][] queries, Path baseFile)
      throws IOException {
    int nDocs = 0;
    int dims = -1;
    for (Leaf lf : leaves) {
      nDocs += lf.size;
      dims = lf.fvv.dimension();
    }
    return "GT-SIG-v1;nDocs=" + nDocs + ";dims=" + dims + ";K=" + K
        + ";nq=" + queries.length + ";base=" + fileSig(baseFile);
  }

  /** Size + head/tail SHA-256 prefix of a file (cheap proxy for content identity). */
  static String fileSig(Path p) throws IOException {
    long size = Files.size(p);
    byte[] head = new byte[65536];
    byte[] tail = new byte[65536];
    int hn = 0;
    int tn = 0;
    try (InputStream in = Files.newInputStream(p)) {
      hn = in.readNBytes(head, 0, head.length);
      long remaining = size - head.length;
      while (remaining > 0) {
        long skipped = in.skip(remaining);
        if (skipped <= 0) {
          break;
        }
        remaining -= skipped;
      }
      tn = in.readNBytes(tail, 0, tail.length);
    }
    MessageDigest md;
    try {
      md = MessageDigest.getInstance("SHA-256");
    } catch (java.security.NoSuchAlgorithmException e) {
      throw new RuntimeException(e);   // SHA-256 is mandated on every JVM
    }
    md.update(Long.toString(size).getBytes(StandardCharsets.UTF_8));
    md.update(head, 0, hn);
    md.update(tail, 0, tn);
    return HexFormat.of().formatHex(md.digest(), 0, 8);
  }

  static float[][] loadFvecs(Path p, int maxn) throws IOException {
    ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
    List<float[]> rows = new ArrayList<>();
    while (buf.remaining() >= 4 && rows.size() < maxn) {
      int dim = buf.getInt();
      float[] v = new float[dim];
      for (int i = 0; i < dim; i++) {
        v[i] = buf.getFloat();
      }
      rows.add(v);
    }
    return rows.toArray(new float[0][]);
  }

  static int[][] loadIvecsTopK(Path p, int maxn, int k) throws IOException {
    ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
    List<int[]> rows = new ArrayList<>();
    while (buf.remaining() >= 4 && rows.size() < maxn) {
      int kk = buf.getInt();
      int[] r = new int[Math.min(k, kk)];
      for (int i = 0; i < kk; i++) {
        int v = buf.getInt();
        if (i < r.length) {
          r[i] = v;
        }
      }
      rows.add(r);
    }
    return rows.toArray(new int[0][]);
  }
}
