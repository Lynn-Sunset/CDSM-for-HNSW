package phase0;

import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Build-quality experiment: does the production graph (M=16, efC=128) starve
 * the search budget, and do better construction parameters close the gap?
 *
 * <p>Rebuilds HNSW graphs from the SAME real vectors with several (M, efC)
 * configs using Lucene's real HnswGraphBuilder, then evaluates identical
 * queries with the instrumented search at budgets 100/200.
 *
 * <p>Usage: Phase1Build &lt;indexDir&gt; &lt;field&gt; &lt;leafDocBase&gt; &lt;numQueries&gt; &lt;tag&gt;
 */
public final class Phase1Build {

  static final int K = 10;

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 5) {
      System.err.println("usage: Phase1Build <indexDir> <field> <leafDocBase> <numQueries> <tag>");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int leafDocBase = Integer.parseInt(args[2]);
    int numQueries = Integer.parseInt(args[3]);
    String tag = args[4];

    System.out.printf("[phase1-build] indexDir=%s field=%s leafDocBase=%d queries=%d tag=%s%n",
        indexDir, field, leafDocBase, numQueries, tag);

    try (DirectoryReader ir = DirectoryReader.open(FSDirectory.open(indexDir, NoLockFactory.INSTANCE))) {
      FSDirectory dir = (FSDirectory) ir.directory();
      LeafReaderContext ctx = null;
      for (LeafReaderContext l : ir.leaves()) {
        if (l.docBase == leafDocBase) {
          ctx = l;
          break;
        }
      }
      if (ctx == null) {
        throw new IllegalStateException("leaf docBase=" + leafDocBase + " not found");
      }
      CodecReader cr = (CodecReader) ctx.reader();
      FloatVectorValues fvv = cr.getFloatVectorValues(field);
      int size = fvv.size();
      float[][] vecs = Phase0Real.loadVectors(fvv, size, fvv.dimension());
      System.out.printf("[leaf] size=%d dim=%d%n", size, vecs[0].length);

      // queries (fixed, same across all graphs)
      Random rnd = new Random(555L * leafDocBase + size);
      int[] qOrds = new int[numQueries];
      for (int i = 0; i < numQueries; i++) {
        qOrds[i] = rnd.nextInt(size);
      }
      // ground truth (shared)
      int[][] ground = new int[numQueries][];
      for (int i = 0; i < numQueries; i++) {
        ground[i] = Phase0Real.bruteForceTopK(vecs, Phase0Real.normalize(vecs[qOrds[i]]), K, null);
      }

      // ---- production graph (M=16 efC=128) ----
      HnswGraph prod = Phase0Real.getRealGraph(dir, cr, field);
      evalGraph("production(M16,efC128)", prod, vecs, qOrds, ground);

      // ---- rebuilds ----
      for (int[] cfg : new int[][]{{16, 128}, {32, 200}, {48, 300}}) {
        int m = cfg[0], efc = cfg[1];
        long t0 = System.currentTimeMillis();
        RandomVectorScorerSupplier sup = new CosineSupplier(vecs);
        HnswGraphBuilder b = HnswGraphBuilder.create(sup, m, efc, 42);
        OnHeapHnswGraph g = b.build(size);
        long ms = System.currentTimeMillis() - t0;
        evalGraph(String.format(Locale.ROOT, "rebuild(M%d,efC%d) in %ds", m, efc, ms / 1000),
            g, vecs, qOrds, ground);
      }
    }
  }

  static void evalGraph(String label, HnswGraph graph, float[][] vecs,
                        int[] qOrds, int[][] ground) throws IOException {
    int size = vecs.length;
    double rec100 = 0, rec200 = 0;
    int descTop10 = 0;
    for (int i = 0; i < qOrds.length; i++) {
      float[] qNorm = Phase0Real.normalize(vecs[qOrds[i]]);
      InstrumentedSimSearch.ScorerFn scorer = ord -> Phase0Real.dot(qNorm, vecs[ord]);
      InstrumentedSimSearch.Result r1 =
          InstrumentedSimSearch.search(graph, size, scorer, K, 100, null);
      rec100 += Phase0Real.recallAt(ground[i], r1.topK);
      InstrumentedSimSearch.Result r2 =
          InstrumentedSimSearch.search(graph, size, scorer, K, 200, null);
      rec200 += Phase0Real.recallAt(ground[i], r2.topK);
      if (Phase0Real.contains(ground[i], r1.descendedNode)) {
        descTop10++;
      }
    }
    int n = qOrds.length;
    double avgDeg0 = avgDegree(graph, 0);
    System.out.printf(Locale.ROOT,
        "%-28s recall@100=%.3f  recall@200=%.3f  descentTop10=%.1f%%  levels=%d  avgDegL0=%.1f%n",
        label, rec100 / n, rec200 / n, 100.0 * descTop10 / n, graph.numLevels(), avgDeg0);
  }

  static double avgDegree(HnswGraph graph, int level) throws IOException {
    int cnt = 0;
    long deg = 0;
    var it = graph.getNodesOnLevel(level);
    while (it.hasNext()) {
      int node = it.nextInt();
      graph.seek(level, node);
      int d = 0;
      while (graph.nextNeighbor() != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS) {
        d++;
      }
      deg += d;
      cnt++;
    }
    return cnt == 0 ? 0 : (double) deg / cnt;
  }

  /** Cosine scorer over normalized stored vectors (higher = closer). */
  static final class CosineSupplier implements RandomVectorScorerSupplier {
    private final float[][] vecs;

    CosineSupplier(float[][] vecs) {
      this.vecs = vecs;
    }

    @Override
    public RandomVectorScorer scorer(int node) {
      final float[] q = vecs[node];
      return new RandomVectorScorer() {
        @Override
        public float score(int ord) {
          return Phase0Real.dot(q, vecs[ord]);
        }

        @Override
        public int maxOrd() {
          return vecs.length - 1;
        }
      };
    }

    @Override
    public RandomVectorScorerSupplier copy() {
      return new CosineSupplier(vecs);
    }
  }
}
