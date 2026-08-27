package phase0;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
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
 * Build a REAL Lucene 9.12 index for GIST1M (1M x 960-d float32, EUCLIDEAN).
 * Default codec = Lucene99HnswVectorsFormat with maxConn=16 / beamWidth=100 —
 * exactly our graph params (M=16, efC=100). Produces the deployment-grade mmap
 * index consumed by PhaseGISTEndToEnd.
 *
 * <p>Usage: PhaseGISTIndex &lt;gistDir&gt; &lt;indexDir&gt;
 */
public final class PhaseGISTIndex {

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 2) {
      System.err.println("usage: PhaseGISTIndex <gistDir> <indexDir>");
      System.exit(2);
    }
    Path gistDir = Paths.get(args[0]);
    Path indexDir = Paths.get(args[1]);
    Files.createDirectories(indexDir);
    System.out.printf("[gist-index] gistDir=%s indexDir=%s%n", gistDir, indexDir);

    if (args.length >= 3 && args[2].equals("merge-only")) {
      try (IndexWriter w = new IndexWriter(FSDirectory.open(indexDir), new IndexWriterConfig())) {
        w.forceMerge(1);
        w.commit();
      }
      long bytes = 0;
      try (var s = Files.walk(indexDir)) {
        for (Path p : s.filter(Files::isRegularFile).toList()) {
          bytes += Files.size(p);
        }
      }
      System.out.printf("[gist-index] merge-only done, index size = %.2f GiB%n",
          bytes / 1073741824.0);
      return;
    }

    float[][] base = loadFvecs(gistDir.resolve("gist_base.fvecs"));
    System.out.printf("[gist-index] loaded %d x %d%n", base.length, base[0].length);

    IndexWriterConfig cfg = new IndexWriterConfig();
    cfg.setRAMBufferSizeMB(256);
    cfg.setUseCompoundFile(false);

    long t0 = System.currentTimeMillis();
    try (IndexWriter w = new IndexWriter(FSDirectory.open(indexDir), cfg)) {
      for (int i = 0; i < base.length; i++) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField("vec", base[i]));   // EUCLIDEAN, FLOAT32
        w.addDocument(doc);
        if ((i + 1) % 100_000 == 0) {
          System.out.printf("[gist-index] %d docs (%.0fs)%n", i + 1,
              (System.currentTimeMillis() - t0) / 1000.0);
        }
      }
      w.forceMerge(1);
      w.commit();
      System.out.printf("[gist-index] forceMerged to 1 segment, committed %d docs in %.0fs%n",
          base.length, (System.currentTimeMillis() - t0) / 1000.0);
    }
    long bytes = 0;
    try (var s = Files.walk(indexDir)) {
      for (Path p : s.filter(Files::isRegularFile).toList()) {
        bytes += Files.size(p);
      }
    }
    System.out.printf("[gist-index] index size on disk = %.2f GiB%n", bytes / 1073741824.0);
  }

  /** little-endian fvecs loader (streaming — file exceeds Java array limit). */
  static float[][] loadFvecs(Path p) throws IOException {
    try (java.io.DataInputStream in = new java.io.DataInputStream(
        new java.io.BufferedInputStream(Files.newInputStream(p), 1 << 20))) {
      List<float[]> rows = new ArrayList<>(1_000_000);
      while (true) {
        int dim;
        try {
          dim = Integer.reverseBytes(in.readInt());
        } catch (java.io.EOFException eof) {
          break;
        }
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
          v[i] = Float.intBitsToFloat(Integer.reverseBytes(in.readInt()));
        }
        rows.add(v);
      }
      return rows.toArray(new float[0][]);
    }
  }
}
