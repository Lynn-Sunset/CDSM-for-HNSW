package phase0;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Build a REAL Lucene 9.12 index for the LAION-CLIP slice (N x 768-d float32,
 * COSINE). Default codec = Lucene99HnswVectorsFormat (maxConn=16, beamWidth=100).
 *
 * <p>Usage: PhaseLAIONIndex &lt;fvecsFile&gt; &lt;indexDir&gt;
 */
public final class PhaseLAIONIndex {

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 2) {
      System.err.println("usage: PhaseLAIONIndex <fvecsFile> <indexDir>");
      System.exit(2);
    }
    Path fvecs = Paths.get(args[0]);
    Path indexDir = Paths.get(args[1]);
    Files.createDirectories(indexDir);
    System.out.printf("[laion-index] fvecs=%s indexDir=%s%n", fvecs, indexDir);

    float[][] base = loadFvecs(fvecs);
    System.out.printf("[laion-index] loaded %d x %d%n", base.length, base[0].length);

    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setRAMBufferSizeMB(256);
    cfg.setUseCompoundFile(false);

    long t0 = System.currentTimeMillis();
    try (IndexWriter w = new IndexWriter(FSDirectory.open(indexDir), cfg)) {
      for (int i = 0; i < base.length; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("vec", base[i], VectorSimilarityFunction.COSINE));
        w.addDocument(doc);
        if ((i + 1) % 50_000 == 0) {
          System.out.printf("[laion-index] %d docs (%.0fs)%n", i + 1,
              (System.currentTimeMillis() - t0) / 1000.0);
        }
      }
      w.forceMerge(1);
      w.commit();
      System.out.printf("[laion-index] forceMerged + committed %d docs in %.0fs%n", base.length,
          (System.currentTimeMillis() - t0) / 1000.0);
    }
    long bytes = 0;
    try (var s = Files.walk(indexDir)) {
      for (Path p : s.filter(Files::isRegularFile).toList()) {
        bytes += Files.size(p);
      }
    }
    System.out.printf("[laion-index] index size on disk = %.2f GiB%n", bytes / 1073741824.0);
  }

  /** little-endian fvecs loader. */
  static float[][] loadFvecs(Path p) throws IOException {
    ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
    List<float[]> rows = new ArrayList<>();
    while (buf.remaining() >= 4) {
      int dim = buf.getInt();
      float[] v = new float[dim];
      for (int i = 0; i < dim; i++) {
        v[i] = buf.getFloat();
      }
      rows.add(v);
    }
    return rows.toArray(new float[0][]);
  }
}
