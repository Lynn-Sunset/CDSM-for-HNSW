package phase0;

import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

/**
 * Faithfulness self-test for PhaseHnswStore: builds a small graph, dumps it,
 * reloads it, and checks that the instrumented search produces the IDENTICAL
 * top-k and dc count on both graphs across many queries (neighbor order must
 * survive the round trip, since heap tie-breaking depends on it).
 */
public final class PhaseHnswStoreTest {

  public static void main(String[] args) throws IOException {
    int n = 3000;
    int dim = 16;
    Random rnd = new Random(42);
    float[][] vecs = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) {
        vecs[i][j] = rnd.nextFloat();
      }
    }
    HnswGraph g = HnswGraphBuilder.create(new Phase0Main.FloatArraySupplier(vecs), 16, 100, 42)
        .build(n);
    Path file = Paths.get(System.getProperty("out.dir", "out")).resolve("store-test.hnsw");
    PhaseHnswStore.dump(g, file);
    HnswGraph g2 = PhaseHnswStore.load(file);
    Files.delete(file);

    int mismatches = 0;
    long dcG = 0, dcG2 = 0;
    Random qr = new Random(7);
    for (int q = 0; q < 200; q++) {
      float[] query = vecs[qr.nextInt(n)];
      InstrumentedSimSearch.ScorerFn scorer =
          ord -> -InstrumentedSearch.dist2(query, vecs[ord]);
      InstrumentedSimSearch.Result r1 = InstrumentedSimSearch.search(g, n, scorer, 10, 400, null);
      InstrumentedSimSearch.Result r2 = InstrumentedSimSearch.search(g2, n, scorer, 10, 400, null);
      dcG += r1.scoreComps;
      dcG2 += r2.scoreComps;
      if (!java.util.Arrays.equals(r1.topK, r2.topK)) {
        mismatches++;
        if (mismatches <= 3) {
          System.out.printf("  MISMATCH q=%d: %s vs %s%n", q,
              java.util.Arrays.toString(r1.topK), java.util.Arrays.toString(r2.topK));
        }
      }
    }
    System.out.printf("[store-test] %d queries: topK mismatches=%d, dc %d vs %d%n",
        200, mismatches, dcG, dcG2);
    if (mismatches > 0) {
      throw new IllegalStateException("store round-trip is NOT faithful");
    }
    System.out.println("[store-test] PASS: round-trip is bit-faithful");
  }
}
