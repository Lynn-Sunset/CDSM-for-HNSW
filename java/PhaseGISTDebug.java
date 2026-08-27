package phase0;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Debug: pinpoint why end-to-end recall is ~0.02 instead of ~0.9. */
public final class PhaseGISTDebug {
  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    Path indexDir = Paths.get(args[0]);
    Path gistDir = Paths.get(args[1]);

    float[][] queries = loadFvecs(gistDir.resolve("gist_query.fvecs"), 1000);
    int[][] gt = loadIvecsTopK(gistDir.resolve("gist_groundtruth.ivecs"), 1000, 10);
    System.out.printf("[dbg] q0 dim=%d q0[0..4]=%s%n", queries[0].length,
        Arrays.toString(Arrays.copyOf(queries[0], 5)));
    System.out.printf("[dbg] gt[0]=%s%n", Arrays.toString(gt[0]));

    try (DirectoryReader ir = DirectoryReader.open(FSDirectory.open(indexDir, NoLockFactory.INSTANCE))) {
      CodecReader cr = (CodecReader) ir.leaves().get(0).reader();
      FloatVectorValues fvv = cr.getFloatVectorValues("vec");
      System.out.printf("[dbg] leaf size=%d dim=%d rav?=%b%n", fvv.size(), fvv.dimension(),
          fvv instanceof RandomAccessVectorValues);
      RandomAccessVectorValues rav = fvv instanceof RandomAccessVectorValues r ? r : null;
      if (rav == null) {
        System.out.println("[dbg] NOT RAV — fallback path");
        return;
      }
      System.out.printf("[dbg] ordToDoc(0)=%d ordToDoc(1)=%d ordToDoc(999999)=%d%n",
          rav.ordToDoc(0), rav.ordToDoc(1), rav.ordToDoc(999999));

      // brute-force top-10 by score via the real scorer
      RandomVectorScorer rs = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
          VectorSimilarityFunction.EUCLIDEAN, rav, queries[0]);
      long t0 = System.nanoTime();
      int[] bestOrd = new int[10];
      float[] bestScore = new float[10];
      Arrays.fill(bestScore, Float.NEGATIVE_INFINITY);
      for (int ord = 0; ord < fvv.size(); ord++) {
        float s = rs.score(ord);
        if (s > bestScore[9]) {
          bestScore[9] = s;
          bestOrd[9] = ord;
          for (int i = 8; i >= 0; i--) {
            if (bestScore[i] < bestScore[i + 1]) {
              float ts = bestScore[i]; bestScore[i] = bestScore[i + 1]; bestScore[i + 1] = ts;
              int to = bestOrd[i]; bestOrd[i] = bestOrd[i + 1]; bestOrd[i + 1] = to;
            } else {
              break;
            }
          }
        }
      }
      long t1 = System.nanoTime();
      int[] topDocs = new int[10];
      for (int i = 0; i < 10; i++) {
        topDocs[i] = rav.ordToDoc(bestOrd[i]);
      }
      Set<Integer> g = new HashSet<>();
      for (int x : gt[0]) g.add(x);
      int hits = 0;
      for (int d : topDocs) if (g.contains(d)) hits++;
      System.out.printf("[dbg] brute-force(real scorer, %.1fs): topDocs=%s hits=%d/10 score0=%.4f%n",
          (t1 - t0) / 1e9, Arrays.toString(topDocs), hits, bestScore[0]);
      // probes: what vectors sit at the TRUE neighbors' ords in the index?
      System.out.printf("[dbg] index score(ord 786559)=%.4f  score(ord 624520)=%.4f  "
              + "score(ord 726827)=%.4f  score(ord 658871)=%.4f%n",
          rs.score(786559), rs.score(624520), rs.score(726827), rs.score(658871));

      // independent brute-force via the REAL production VectorScorer path
      // (fvv.scorer — doc-id iteration, Panama kernels)
      org.apache.lucene.search.VectorScorer vs = fvv.scorer(queries[0]);
      var vit = vs.iterator();
      int[] bestDoc = new int[10];
      float[] bestS = new float[10];
      Arrays.fill(bestS, Float.NEGATIVE_INFINITY);
      for (int doc = vit.nextDoc(); doc != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
           doc = vit.nextDoc()) {
        float s = vs.score();
        if (s > bestS[9]) {
          bestS[9] = s;
          bestDoc[9] = doc;
          for (int i = 8; i >= 0; i--) {
            if (bestS[i] < bestS[i + 1]) {
              float ts = bestS[i]; bestS[i] = bestS[i + 1]; bestS[i + 1] = ts;
              int to = bestDoc[i]; bestDoc[i] = bestDoc[i + 1]; bestDoc[i + 1] = to;
            } else {
              break;
            }
          }
        }
      }
      int hits3 = 0;
      for (int i = 0; i < 10; i++) {
        if (g.contains(bestDoc[i])) hits3++;
      }
      System.out.printf("[dbg] brute-force(VectorScorer path): topDocs=%s hits=%d/10%n",
          Arrays.toString(bestDoc), hits3);
      // compare the two independent brute-forces (scorer vs VectorScorer)
      Set<Integer> s1 = new HashSet<>();
      for (int d : topDocs) s1.add(d);
      int agree = 0;
      for (int d : bestDoc) if (s1.contains(d)) agree++;
      System.out.printf("[dbg] agreement(RAV-scorer vs VectorScorer top-10) = %d/10%n", agree);
    }
  }

  static float[][] loadFvecs(Path p, int maxn) throws IOException {
    ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
    List<float[]> rows = new java.util.ArrayList<>();
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
    List<int[]> rows = new java.util.ArrayList<>();
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
