package phase0;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
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
 * Phase 0 on REAL production data: reads the actual HNSW graphs and vectors
 * from an Elasticsearch index's Lucene segments (off-heap, no rebuild), runs
 * the instrumented search, computes per-segment ground truth, classifies
 * queries, evaluates restart oracles, and cross-validates three ways:
 *   (a) instrumented port vs the REAL production search path
 *       (KnnVectorsReader.search + counting collector) on the same segment;
 *   (b) merged top-k vs the ES REST API baseline;
 *   (c) both vs exact brute force.
 *
 * <p>Usage: Phase0Real &lt;indexDir&gt; &lt;field&gt; &lt;numQueries&gt; &lt;visitLimit&gt;
 * &lt;outTag&gt; [esQueryFile] [esTopIdsFile]
 */
public final class Phase0Real {

  static final int K = 10;

  /** One candidate for the global cross-leaf merge. */
  record Cand(int leafIdx, int docBase, int ord, float score) {}

  public static void main(String[] args) throws IOException {
    // must run before any Lucene/ES codec class is loaded: ES SPI codec
    // providers require a logging backend (see EsLoggingBootstrap)
    EsLoggingBootstrap.install();

    if (args.length < 5) {
      System.err.println("usage: Phase0Real <indexDir> <field> <numQueries> <visitLimit> <outTag> [esQueryFile] [esTopIdsFile]");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int numQueries = Integer.parseInt(args[2]);
    int visitLimit = Integer.parseInt(args[3]);
    String tag = args[4];
    Path esQueryFile = args.length >= 6 ? Paths.get(args[5]) : null;
    Path esTopIdsFile = args.length >= 7 ? Paths.get(args[6]) : null;

    System.out.printf("[phase0-real] indexDir=%s field=%s queries=%d visitLimit=%d tag=%s%n",
        indexDir, field, numQueries, visitLimit, tag);

    try (DirectoryReader ir = DirectoryReader.open(FSDirectory.open(indexDir, NoLockFactory.INSTANCE))) {
      FSDirectory dir = (FSDirectory) ir.directory();
      List<LeafReaderContext> leaves = ir.leaves();
      System.out.printf("[phase0-real] %d leaves%n", leaves.size());

      Agg agg = new Agg();
      int totalQueries = 0;

      for (LeafReaderContext ctx : leaves) {
        CodecReader cr = (CodecReader) ctx.reader();
        FloatVectorValues fvv = cr.getFloatVectorValues(field);
        if (fvv == null) {
          continue;
        }
        int size = fvv.size();
        int dim = fvv.dimension();
        if (size < 100) {
          System.out.printf("[leaf] size=%d too small, skip%n", size);
          continue;
        }

        float[][] vecs = loadVectors(fvv, size, dim);
        HnswGraph graph = getRealGraph(dir, cr, field);
        List<Integer> topPool = collectTopLevel(graph);
        System.out.printf("[leaf docBase=%d] size=%d dim=%d levels=%d entry=%d topPool=%d deleted=%b%n",
            ctx.docBase, size, dim, graph.numLevels(), graph.entryNode(), topPool.size(),
            ctx.reader().getLiveDocs() != null);

        int qPerLeaf = Math.max(1, numQueries / leaves.size());
        totalQueries += qPerLeaf;
        processLeaf(ctx, field, size, vecs, graph, topPool, qPerLeaf, visitLimit, agg);
      }

      System.out.printf("%n==== Phase0-real aggregate report (tag=%s field=%s, %d leaves, %d queries) ====%n",
          tag, field, leaves.size(), totalQueries);
      agg.report(System.out);

      // cross-validation against ES baseline + production search path
      if (esQueryFile != null) {
        crossValidate(ir, field, esQueryFile, esTopIdsFile, visitLimit);
      }

      // per-query CSV
      Path outDir = Paths.get(System.getProperty("out.dir", "out"));
      Files.createDirectories(outDir);
      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
          outDir.resolve("real-per-query-" + tag + ".csv"), StandardCharsets.UTF_8))) {
        pw.println("qid,size,D,patience,ratio,recall,class,gapRatio,descendedInTopK,oracleRecall,twoSeedRecall");
        for (String line : agg.csvLines) {
          pw.println(line);
        }
      }
      System.out.printf("[phase0-real] wrote out/real-per-query-%s.csv%n", tag);
    }
  }

  // ---------- real graph acquisition (through the per-field wrapper) ----------

  /**
   * The codec reader returned by {@link CodecReader#getVectorReader()} is the
   * PerFieldKnnVectorsFormat wrapper which does NOT expose the graph. Reach
   * through it: read the concrete per-field format name from the FieldInfo
   * attribute, instantiate that format, and open a standalone reader for this
   * segment so {@link HnswGraphProvider#getGraph(String)} works.
   */
  static HnswGraph getRealGraph(FSDirectory dir, CodecReader cr, String field) throws IOException {
    FieldInfo fi = cr.getFieldInfos().fieldInfo(field);
    String fmtName = fi.getAttribute(PerFieldKnnVectorsFormat.PER_FIELD_FORMAT_KEY);
    if (fmtName == null) {
      fmtName = "Lucene99HnswVectorsFormat";
    }
    KnnVectorsFormat fmt = KnnVectorsFormat.forName(fmtName);
    SegmentReader sr = (SegmentReader) cr;
    SegmentCommitInfo sci = sr.getSegmentInfo();
    org.apache.lucene.index.SegmentInfo si = sci.info;
    IOContext ctx = IOContext.READ;
    // compound files: all per-field sub-files live inside the .cfs; a
    // standalone reader must be opened against the compound directory
    org.apache.lucene.store.Directory d = si.getUseCompoundFile()
        ? si.getCodec().compoundFormat().getCompoundReader(dir, si, ctx)
        : dir;

    // Auto-detect the per-field file suffix. ES writes per-field files as
    // "<seg>_<formatName>_<counter>.<ext>": the FieldInfo stores the counter
    // under PER_FIELD_SUFFIX_KEY, so the real suffix is formatName + "_" +
    // counter. Fall back to scanning the directory only when the attribute
    // is absent.
    String counter = fi.getAttribute(PerFieldKnnVectorsFormat.PER_FIELD_SUFFIX_KEY);
    String suffix = counter != null ? fmtName + "_" + counter : null;
    if (suffix == null) {
      for (String f : d.listAll()) {
        if (f.endsWith(".vemf") && f.startsWith(si.name + "_")) {
          suffix = f.substring(si.name.length() + 1, f.length() - ".vemf".length());
        }
      }
    }
    SegmentReadState state = suffix != null
        ? new SegmentReadState(d, si, cr.getFieldInfos(), ctx, suffix)
        : new SegmentReadState(d, si, cr.getFieldInfos(), ctx);
    KnnVectorsReader real = fmt.fieldsReader(state);
    if (!(real instanceof HnswGraphProvider provider)) {
      throw new IllegalStateException("per-field sub-reader " + real.getClass()
          + " is not an HnswGraphProvider");
    }
    System.out.printf("[graph] field=%s per-field format=%s suffix=%s compound=%b reader=%s%n",
        field, fmtName, suffix, si.getUseCompoundFile(), real.getClass().getSimpleName());
    HnswGraph g = provider.getGraph(field);
    if (g.numLevels() == 0) {
      System.out.printf("[graph] WARNING: empty graph for field=%s suffix=%s — likely wrong per-field suffix%n",
          field, suffix);
    }
    return g;
  }

  // ---------- per-leaf pipeline ----------

  static void processLeaf(LeafReaderContext ctx, String field, int size, float[][] vecs,
                          HnswGraph graph, List<Integer> topPool, int numQueries,
                          int visitLimit, Agg agg) throws IOException {
    org.apache.lucene.util.Bits liveDocs = ctx.reader().getLiveDocs();
    RandomAccessVectorValues rav = null;
    try {
      FloatVectorValues fvv2 = ((CodecReader) ctx.reader()).getFloatVectorValues(field);
      rav = fvv2 instanceof RandomAccessVectorValues r ? r : null;
    } catch (IOException e) {
      // fall through with identity mapping
    }
    final RandomAccessVectorValues ravF = rav;
    org.apache.lucene.util.Bits accept = (liveDocs != null && ravF != null) ? new org.apache.lucene.util.Bits() {
      @Override
      public boolean get(int ord) {
        return liveDocs.get(ravF.ordToDoc(ord));
      }

      @Override
      public int length() {
        return size;
      }
    } : null;

    Random rnd = new Random(42L * ctx.docBase + size);
    for (int q = 0; q < numQueries; q++) {
      int qOrd = rnd.nextInt(size);
      float[] qv = vecs[qOrd];
      float[] qNorm = normalize(qv);

      InstrumentedSimSearch.ScorerFn scorer = ord -> dot(qNorm, vecs[ord]);

      InstrumentedSimSearch.Result r =
          InstrumentedSimSearch.search(graph, size, scorer, K, visitLimit, accept);
      int[] ground = bruteForceTopK(vecs, qNorm, K, accept);
      double recall = recallAt(ground, r.topK);
      String cls = recall >= 0.9 ? "GOOD" : recall <= 0.3 ? "BAD_BASIN" : "PARTIAL";

      float trueBest = dot(qNorm, vecs[ground[0]]);
      float foundBest = r.bestScore();
      double gapRatio = foundBest <= 0 ? Double.POSITIVE_INFINITY : trueBest / foundBest;
      boolean descendedInTopK = contains(ground, r.descendedNode);

      double oracleRecall = -1, twoSeedRecall = -1;
      if (!cls.equals("GOOD") && topPool.size() >= 2) {
        int alt = farthestFromEntry(topPool, r.entryNode, vecs);
        InstrumentedSimSearch.Result r2 =
            InstrumentedSimSearch.search(graph, size, scorer, K, visitLimit, accept, alt);
        oracleRecall = recallAt(ground, r2.topK);
        InstrumentedSimSearch.Result r3 = InstrumentedSimSearch.searchTwoSeed(
            graph, size, scorer, K, visitLimit, accept, r.entryNode, alt);
        twoSeedRecall = recallAt(ground, r3.topK);
      }

      agg.add(ctx.docBase * 10000 + q, size, r, recall, cls, gapRatio,
          descendedInTopK, oracleRecall, twoSeedRecall);
    }
  }

  // ---------- cross-validation ----------

  static void crossValidate(DirectoryReader ir, String field, Path queryFile,
                            Path topIdsFile, int visitLimit) throws IOException {
    String line = Files.readString(queryFile).trim();
    String[] parts = line.split(",");
    float[] q = new float[parts.length];
    for (int i = 0; i < parts.length; i++) {
      q[i] = Float.parseFloat(parts[i].trim());
    }
    float[] qNorm = normalize(q);

    java.util.PriorityQueue<Cand> portBest = new java.util.PriorityQueue<>(K + 1,
        (a, b) -> Float.compare(a.score(), b.score()));
    java.util.PriorityQueue<Cand> prodBest = new java.util.PriorityQueue<>(K + 1,
        (a, b) -> Float.compare(a.score(), b.score()));
    java.util.PriorityQueue<Cand> trueBest = new java.util.PriorityQueue<>(K + 1,
        (a, b) -> Float.compare(a.score(), b.score()));

    long portD = 0, prodD = 0;
    int portPat = 0, prodPat = 0;

    List<LeafReaderContext> leaves = ir.leaves();
    for (int li = 0; li < leaves.size(); li++) {
      LeafReaderContext ctx = leaves.get(li);
      CodecReader cr = (CodecReader) ctx.reader();
      FloatVectorValues fvv = cr.getFloatVectorValues(field);
      if (fvv == null) {
        continue;
      }
      float[][] vecs = loadVectors(fvv, fvv.size(), fvv.dimension());
      HnswGraph graph = getRealGraph((FSDirectory) ir.directory(), cr, field);

      org.apache.lucene.util.Bits liveDocs = ctx.reader().getLiveDocs();
      RandomAccessVectorValues rav = fvv instanceof RandomAccessVectorValues r ? r : null;
      int leafSize = fvv.size();
      org.apache.lucene.util.Bits accept = (liveDocs != null && rav != null)
          ? new org.apache.lucene.util.Bits() {
            @Override
            public boolean get(int ord) {
              return liveDocs.get(rav.ordToDoc(ord));
            }

            @Override
            public int length() {
              return leafSize;
            }
          } : null;

      // (1) instrumented port
      float[][] vv = vecs;
      InstrumentedSimSearch.ScorerFn scorer = ord -> dot(qNorm, vv[ord]);
      InstrumentedSimSearch.Result r =
          InstrumentedSimSearch.search(graph, leafSize, scorer, K, visitLimit, accept);
      portD += r.scoreComps;
      portPat += r.patience();
      for (int i = 0; i < r.topK.length; i++) {
        int did = rav != null ? rav.ordToDoc(r.topK[i]) : r.topK[i];
        offer(portBest, new Cand(li, ctx.docBase, did, r.topKScores[i]), K);
      }

      // (2) REAL production search path: codec reader search + counting collector
      //     (live docs passed exactly as Lucene KnnVectorQuery does)
      CountingCollector prod = new CountingCollector(K, visitLimit);
      cr.getVectorReader().search(field, q, prod, liveDocs);
      prodD += prod.visitedCount();
      prodPat += prod.patience();
      int[] prodOrds = prod.ords();
      float[] prodScores = prod.scores();
      for (int i = 0; i < prodOrds.length; i++) {
        offer(prodBest, new Cand(li, ctx.docBase, prodOrds[i], prodScores[i]), K);
      }

      // (3) exact brute force on this leaf (deleted docs excluded)
      int[] gt = bruteForceTopK(vecs, qNorm, K, accept);
      for (int i = 0; i < gt.length; i++) {
        int did = rav != null ? rav.ordToDoc(gt[i]) : gt[i];
        offer(trueBest, new Cand(li, ctx.docBase, did, dot(qNorm, vecs[gt[i]])), K);
      }
    }

    List<Cand> portTop = drain(portBest);
    List<Cand> prodTop = drain(prodBest);
    List<Cand> trueTop = drain(trueBest);

    System.out.printf("%n[ES cross-validation] field=%s visitLimit=%d%n", field, visitLimit);
    List<String> esIds = topIdsFile != null && Files.exists(topIdsFile)
        ? Files.readAllLines(topIdsFile) : List.of();

    // debug: what stored field names does doc 0 carry?
    try {
      System.out.println("[debug] stored fields of doc 0: "
          + ir.document(0).getFields().stream().map(f -> f.name()).toList()
          + " _id=" + ir.document(0).get("_id"));
      System.out.println("[debug] stored fields of doc 68: "
          + ir.document(68).getFields().stream().map(f -> f.name()).toList()
          + " _id=" + ir.document(68).get("_id"));
      System.out.println("[debug] stored fields of doc 12: "
          + ir.document(12).getFields().stream().map(f -> f.name()).toList()
          + " _id=" + ir.document(12).get("_id"));
    } catch (Exception e) {
      System.out.println("[debug] document() failed: " + e);
    }

    System.out.printf("%-4s %-26s %-26s %-26s %-26s%n",
        "rank", "ES_api(_id)", "port(_id)", "prod(_id)", "true_bf(_id)");
    Set<String> portSet = new HashSet<>(), prodSet = new HashSet<>(),
        trueSet = new HashSet<>(), esSet = new HashSet<>(esIds);
    int esDenom = Math.min(K, esIds.size());
    for (int i = 0; i < K; i++) {
      String esId = i < esIds.size() ? esIds.get(i) : "-";
      String pId = i < portTop.size() ? idOf(ir, portTop.get(i)) : "-";
      String dId = i < prodTop.size() ? idOf(ir, prodTop.get(i)) : "-";
      String tId = i < trueTop.size() ? idOf(ir, trueTop.get(i)) : "-";
      portSet.add(pId);
      prodSet.add(dId);
      trueSet.add(tId);
      System.out.printf(Locale.ROOT, "%-4d %-26s %-26s %-26s %-26s%n",
          i + 1, shorten(esId), shorten(pId), shorten(dId), shorten(tId));
    }
    Set<String> p = new HashSet<>(portSet), d = new HashSet<>(prodSet), t = new HashSet<>(trueSet);
    double jacPP = jaccard(p, d), jacEP = jaccard(esSet, p), jacED = jaccard(esSet, d),
        jacPT = jaccard(p, t);
    System.out.printf(Locale.ROOT,
        "set agreement (top-10 Jaccard): ES vs port=%.2f, ES vs prod=%.2f, port vs prod=%.2f, port vs brute=%.2f%n",
        jacEP, jacED, jacPP, jacPT);
    System.out.printf(Locale.ROOT,
        "cost (across leaves): port D=%d patience=%d | prod D=%d patience=%d%n",
        portD, portPat, prodD, prodPat);
  }

  static double jaccard(Set<String> a, Set<String> b) {
    if (a.isEmpty() && b.isEmpty()) return 1.0;
    Set<String> inter = new HashSet<>(a);
    inter.retainAll(b);
    Set<String> union = new HashSet<>(a);
    union.addAll(b);
    return union.isEmpty() ? 0 : (double) inter.size() / union.size();
  }

  static List<Cand> drain(java.util.PriorityQueue<Cand> pq) {
    List<Cand> out = new ArrayList<>();
    while (!pq.isEmpty()) {
      out.add(0, pq.poll());
    }
    return out;
  }

  static void offer(java.util.PriorityQueue<Cand> pq, Cand c, int k) {
    pq.offer(c);
    if (pq.size() > k) {
      pq.poll();
    }
  }

  static String idOf(DirectoryReader ir, Cand c) throws IOException {
    try {
      var doc = ir.document(c.docBase() + c.ord());
      org.apache.lucene.util.BytesRef br = doc.getBinaryValue("_source");
      if (br != null) {
        byte[] src = java.util.Arrays.copyOfRange(br.bytes, br.offset, br.offset + br.length);
        try (org.elasticsearch.xcontent.XContentParser p =
                 org.elasticsearch.xcontent.json.JsonXContent.jsonXContent
                     .createParser(org.elasticsearch.xcontent.XContentParserConfiguration.EMPTY, src)) {
          java.util.Map<String, Object> map = p.map();
          Object id = map.get("id");
          if (id != null) {
            return id.toString();
          }
        }
      }
      String id = doc.get("_id");
      return id == null ? "#no-id@doc=" + (c.docBase() + c.ord()) : id;
    } catch (Exception e) {
      return "#err@" + (c.docBase() + c.ord());
    }
  }

  /** TopKnnCollector with last-improvement counters (visitedCount() is final
   *  on AbstractKnnCollector, so it is read rather than overridden). */
  static final class CountingCollector extends TopKnnCollector {
    long lastImproveVisited = -1;
    final List<Integer> ids = new ArrayList<>();
    final List<Float> sims = new ArrayList<>();

    CountingCollector(int k, int visitedLimit) {
      super(k, visitedLimit);
    }

    @Override
    public boolean collect(int docId, float similarity) {
      boolean ok = super.collect(docId, similarity);
      if (ok) {
        lastImproveVisited = visitedCount();
        ids.add(docId);
        sims.add(similarity);
      }
      return ok;
    }

    long patience() {
      return lastImproveVisited < 0 ? visitedCount() : visitedCount() - lastImproveVisited;
    }

    int[] ords() {
      int[] a = new int[ids.size()];
      for (int i = 0; i < a.length; i++) {
        a[i] = ids.get(i);
      }
      return a;
    }

    float[] scores() {
      float[] a = new float[sims.size()];
      for (int i = 0; i < a.length; i++) {
        a[i] = sims.get(i);
      }
      return a;
    }
  }

  // ---------- helpers ----------

  static float[][] loadVectors(FloatVectorValues fvv, int size, int dim) throws IOException {
    float[][] vecs = new float[size][];
    int ord;
    while ((ord = fvv.nextDoc()) != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS) {
      vecs[ord] = fvv.vectorValue().clone();
    }
    return vecs;
  }

  static List<Integer> collectTopLevel(HnswGraph graph) throws IOException {
    List<Integer> pool = new ArrayList<>();
    var it = graph.getNodesOnLevel(graph.numLevels() - 1);
    while (it.hasNext()) {
      pool.add(it.nextInt());
    }
    return pool;
  }

  static float[] normalize(float[] v) {
    double sum = 0;
    for (float x : v) {
      sum += (double) x * x;
    }
    float n = (float) Math.sqrt(sum);
    float[] out = new float[v.length];
    for (int i = 0; i < v.length; i++) {
      out[i] = v[i] / n;
    }
    return out;
  }

  static float dot(float[] a, float[] b) {
    float s = 0f;
    for (int i = 0; i < a.length; i++) {
      s += a[i] * b[i];
    }
    return s;
  }

  static int[] bruteForceTopK(float[][] vecs, float[] qNorm, int k,
                              org.apache.lucene.util.Bits accept) {
    java.util.PriorityQueue<Integer> pq = new java.util.PriorityQueue<>(k + 1,
        (a, b) -> Float.compare(dot(qNorm, vecs[a]), dot(qNorm, vecs[b])));
    for (int i = 0; i < vecs.length; i++) {
      if (accept != null && !accept.get(i)) {
        continue;   // skip deleted-doc vectors
      }
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

  static double recallAt(int[] ground, int[] found) {
    Set<Integer> g = new HashSet<>(ground.length * 2);
    for (int v : ground) {
      g.add(v);
    }
    int hit = 0;
    for (int v : found) {
      if (g.contains(v)) {
        hit++;
      }
    }
    return (double) hit / ground.length;
  }

  static boolean contains(int[] arr, int v) {
    for (int x : arr) {
      if (x == v) return true;
    }
    return false;
  }

  static int farthestFromEntry(List<Integer> pool, int entry, float[][] vecs) {
    int best = pool.get(0);
    float bestD = -1f;
    for (int v : pool) {
      float d = 0f;
      float[] a = vecs[v], b = vecs[entry];
      for (int i = 0; i < a.length; i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
      }
      if (d > bestD) {
        bestD = d;
        best = v;
      }
    }
    return best;
  }

  static String shorten(String s) {
    return s.length() > 24 ? s.substring(0, 12) + "..." + s.substring(s.length() - 8) : s;
  }

  // ---------- aggregation ----------

  static final class Agg {
    final List<String> csvLines = new ArrayList<>();
    final List<Double> recalls = new ArrayList<>();
    final List<Long> ds = new ArrayList<>();
    final List<Long> pats = new ArrayList<>();
    final List<Double> ratios = new ArrayList<>();
    final List<String> classes = new ArrayList<>();
    int goodCount = 0, badCount = 0, oracleTested = 0, oracleImproved = 0, twoSeedImproved = 0;
    double oracleBefore = 0, oracleAfter = 0, twoSeedAfter = 0;
    int descendedInTopKCount = 0;
    final List<Double> badGap = new ArrayList<>();
    final List<Double> goodGap = new ArrayList<>();

    void add(int qid, int size, InstrumentedSimSearch.Result r,
             double recall, String cls, double gapRatio, boolean descendedInTopK,
             double oracleRecall, double twoSeedRecall) {
      recalls.add(recall);
      ds.add(r.scoreComps);
      pats.add(r.patience());
      ratios.add(r.patienceRatio());
      classes.add(cls);
      if (cls.equals("GOOD")) goodCount++;
      if (cls.equals("BAD_BASIN")) badCount++;
      if (descendedInTopK) descendedInTopKCount++;
      if (cls.equals("GOOD")) goodGap.add(gapRatio); else badGap.add(gapRatio);
      if (!cls.equals("GOOD")) {
        oracleTested++;
        oracleBefore += recall;
        if (oracleRecall >= 0) {
          oracleAfter += oracleRecall;
          if (oracleRecall > recall) oracleImproved++;
        }
        if (twoSeedRecall >= 0) {
          twoSeedAfter += twoSeedRecall;
          if (twoSeedRecall > recall) twoSeedImproved++;
        }
      }
      csvLines.add(String.format(Locale.ROOT,
          "%d,%d,%d,%d,%.4f,%.3f,%s,%.3f,%b,%.3f,%.3f",
          qid, size, r.scoreComps, r.patience(), r.patienceRatio(),
          recall, cls, gapRatio, descendedInTopK, oracleRecall, twoSeedRecall));
    }

    void report(Appendable out) throws IOException {
      int n = recalls.size();
      out.append(String.format(Locale.ROOT, "%n[recall distribution] queries=%d good=%d (%.1f%%) bad_basin=%d (%.1f%%)%n",
          n, goodCount, 100.0 * goodCount / n, badCount, 100.0 * badCount / n));
      out.append(String.format(Locale.ROOT,
          "  mean recall=%.3f  mean D=%.0f  mean patience=%.0f  mean ratio=%.3f%n",
          mean(recalls), mean(ds), mean(pats), mean(ratios)));
      out.append(String.format(Locale.ROOT,
          "  descended-in-ground-truth-topK rate = %.1f%% (descent health)%n",
          100.0 * descendedInTopKCount / n));
      out.append(String.format(Locale.ROOT,
          "  gap ratio trueBest/foundBest: GOOD mean=%.3f  BAD mean=%.3f (>1 = ended short of true best)%n",
          mean(goodGap), mean(badGap)));
      if (oracleTested > 0) {
        out.append(String.format(Locale.ROOT,
            "%n[oracle-R: full re-search from farthest top-level entry] tested=%d improved=%d (%.1f%%) mean recall %.3f -> %.3f%n",
            oracleTested, oracleImproved, 100.0 * oracleImproved / oracleTested,
            oracleBefore / oracleTested, oracleAfter / oracleTested));
        out.append(String.format(Locale.ROOT,
            "[oracle-T: two-seed beam] tested=%d improved=%d (%.1f%%) mean recall %.3f -> %.3f%n",
            oracleTested, twoSeedImproved, 100.0 * twoSeedImproved / oracleTested,
            oracleBefore / oracleTested, twoSeedAfter / oracleTested));
      }
    }

    static double mean(List<? extends Number> xs) {
      double s = 0;
      for (Number x : xs) s += x.doubleValue();
      return s / xs.size();
    }
  }
}
