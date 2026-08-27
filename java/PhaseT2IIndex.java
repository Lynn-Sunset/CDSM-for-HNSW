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

/**
 * Build a REAL Lucene 9.12 index for ann-t2i-1m (1M x 200-d float32, COSINE).
 * Default codec = Lucene99HnswVectorsFormat (maxConn=16, beamWidth=100).
 *
 * <p>Usage: PhaseT2IIndex &lt;dataDir&gt; &lt;indexDir&gt;
 */
public final class PhaseT2IIndex {

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 2) {
      System.err.println("usage: PhaseT2IIndex <dataDir> <indexDir>");
      System.exit(2);
    }
    Path dataDir = Paths.get(args[0]);
    Path indexDir = Paths.get(args[1]);
    Files.createDirectories(indexDir);
    System.out.printf("[t2i-index] dataDir=%s indexDir=%s%n", dataDir, indexDir);

    float[][] base = loadFbin(dataDir.resolve("base.1M.fbin"));
    System.out.printf("[t2i-index] loaded %d x %d%n", base.length, base[0].length);

    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setRAMBufferSizeMB(256);
    cfg.setUseCompoundFile(false);

    long t0 = System.currentTimeMillis();
    try (IndexWriter w = new IndexWriter(FSDirectory.open(indexDir), cfg)) {
      for (int i = 0; i < base.length; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("vec", base[i], VectorSimilarityFunction.COSINE));
        w.addDocument(doc);
        if ((i + 1) % 100_000 == 0) {
          System.out.printf("[t2i-index] %d docs (%.0fs)%n", i + 1,
              (System.currentTimeMillis() - t0) / 1000.0);
        }
      }
      w.forceMerge(1);
      w.commit();
      System.out.printf("[t2i-index] forceMerged + committed %d docs in %.0fs%n", base.length,
          (System.currentTimeMillis() - t0) / 1000.0);
    }
    long bytes = 0;
    try (var s = Files.walk(indexDir)) {
      for (Path p : s.filter(Files::isRegularFile).toList()) {
        bytes += Files.size(p);
      }
    }
    System.out.printf("[t2i-index] index size on disk = %.2f GiB%n", bytes / 1073741824.0);
  }

  /** fbin loader: [int n][int dim][floats], little-endian. */
  static float[][] loadFbin(Path p) throws IOException {
    ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
    int n = buf.getInt();
    int dim = buf.getInt();
    float[][] out = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) {
        out[i][j] = buf.getFloat();
      }
    }
    return out;
  }
}
