package phase0;

import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Phase B: local intrinsic dimensionality (LID) estimation (Levina-Bickel MLE,
 * k=20) per dataset, to relate plateau height to data heterogeneity.
 *
 * <p>Usage: PhaseBLid &lt;indexDir&gt; &lt;field&gt; &lt;leafDocBase&gt; [numSamples]
 *        — real data mode (cosine/L2 on normalized stored vectors).
 *        PhaseBLid hetero — synthetic unbalanced-cluster mode.
 */
public final class PhaseBLid {

  static final int K_NEIGH = 20;

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length >= 1 && args[0].equals("hetero")) {
      heteroLid();
      return;
    }
    if (args.length < 3) {
      System.err.println("usage: PhaseBLid <indexDir> <field> <leafDocBase> [numSamples]");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int leafDocBase = Integer.parseInt(args[2]);
    int numSamples = args.length >= 4 ? Integer.parseInt(args[3]) : 200;

    try (DirectoryReader ir = DirectoryReader.open(FSDirectory.open(indexDir, NoLockFactory.INSTANCE))) {
      LeafReaderContext ctx = null;
      for (LeafReaderContext l : ir.leaves()) {
        if (l.docBase == leafDocBase) {
          ctx = l;
          break;
        }
      }
      if (ctx == null) {
        throw new IllegalStateException("leaf not found: " + leafDocBase);
      }
      CodecReader cr = (CodecReader) ctx.reader();
      FloatVectorValues fvv = cr.getFloatVectorValues(field);
      int size = fvv.size();
      float[][] vecs = Phase0Real.loadVectors(fvv, size, fvv.dimension());
      System.out.printf("[lid] %s/%s leaf=%d size=%d dim=%d%n",
          indexDir.getFileName(), field, leafDocBase, size, vecs[0].length);
      reportLid(vecs, numSamples, "real-" + field);
    }
  }

  static void heteroLid() {
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
    // MEGA subset
    float[][] mega = new float[9600][];
    System.arraycopy(vecs, 0, mega, 0, 9600);
    float[][] small = new float[5 * 480][];
    System.arraycopy(vecs, 9600, small, 0, small.length);
    reportLid(mega, 200, "hetero-MEGA");
    reportLid(small, 200, "hetero-SMALL");
  }

  static void reportLid(float[][] vecs, int numSamples, String label) {
    Random rnd = new Random(1234 + label.hashCode());
    List<Double> lids = new ArrayList<>();
    int n = vecs.length;
    int dim = vecs[0].length;
    float[] dists = new float[n];
    for (int s = 0; s < numSamples; s++) {
      float[] p = vecs[rnd.nextInt(n)];
      for (int i = 0; i < n; i++) {
        float d = 0;
        float[] v = vecs[i];
        for (int j = 0; j < dim; j++) {
          float diff = p[j] - v[j];
          d += diff * diff;
        }
        dists[i] = d;
      }
      java.util.Arrays.sort(dists);
      // d_1 = 0 (self); MLE over d_2..d_k relative to d_k (k = K_NEIGH)
      double sum = 0;
      float dk = dists[K_NEIGH];
      for (int j = 1; j < K_NEIGH; j++) {
        sum += Math.log(dists[j] / dk);
      }
      double lid = -(K_NEIGH - 1) / sum;
      lids.add(lid);
    }
    double mean = 0, var = 0;
    for (double l : lids) {
      mean += l;
    }
    mean /= lids.size();
    for (double l : lids) {
      var += (l - mean) * (l - mean);
    }
    var /= lids.size();
    List<Double> sorted = new ArrayList<>(lids);
    sorted.sort(null);
    double median = sorted.get(sorted.size() / 2);
    System.out.printf(Locale.ROOT,
        "[lid] %-16s n=%d  meanLID=%.1f  medianLID=%.1f  var=%.1f%n",
        label, lids.size(), mean, median, var);
  }
}
