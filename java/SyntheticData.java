package phase0;

import java.util.Random;

/**
 * Multi-cluster Gaussian synthetic data.
 *
 * Cluster centers are drawn uniformly from [-1,1]^dim; members are
 * center + N(0, sigma) per dimension. A fraction of queries are
 * out-of-distribution (uniform over the cube).
 */
public final class SyntheticData {

  public final int dim;
  public final int numClusters;
  public final float[][] vectors;      // N x dim
  public final int[] vectorCluster;    // cluster label per vector
  public final float[][] queries;      // Q x dim
  public final int[] queryCluster;     // cluster label per query, -1 = OOD
  public final float[][] centers;      // cluster centers (for diagnostics)

  private SyntheticData(int dim, int numClusters, float[][] vectors, int[] vectorCluster,
                        float[][] queries, int[] queryCluster, float[][] centers) {
    this.dim = dim;
    this.numClusters = numClusters;
    this.vectors = vectors;
    this.vectorCluster = vectorCluster;
    this.queries = queries;
    this.queryCluster = queryCluster;
    this.centers = centers;
  }

  public static SyntheticData generate(int dim, int numClusters, int perCluster,
                                       int numQueries, long seed, double sigma,
                                       double oodFraction) {
    Random r = new Random(seed);

    float[][] centers = new float[numClusters][dim];
    for (int c = 0; c < numClusters; c++) {
      for (int d = 0; d < dim; d++) {
        centers[c][d] = (float) (r.nextDouble() * 2.0 - 1.0);
      }
    }

    int n = numClusters * perCluster;
    float[][] vecs = new float[n][dim];
    int[] vc = new int[n];
    int idx = 0;
    for (int c = 0; c < numClusters; c++) {
      for (int i = 0; i < perCluster; i++) {
        for (int d = 0; d < dim; d++) {
          vecs[idx][d] = centers[c][d] + (float) (r.nextGaussian() * sigma);
        }
        vc[idx] = c;
        idx++;
      }
    }

    float[][] qs = new float[numQueries][dim];
    int[] qc = new int[numQueries];
    for (int i = 0; i < numQueries; i++) {
      if (r.nextDouble() < oodFraction) {
        for (int d = 0; d < dim; d++) {
          qs[i][d] = (float) (r.nextDouble() * 2.0 - 1.0);
        }
        qc[i] = -1;
      } else {
        int c = r.nextInt(numClusters);
        for (int d = 0; d < dim; d++) {
          qs[i][d] = centers[c][d] + (float) (r.nextGaussian() * sigma);
        }
        qc[i] = c;
      }
    }

    return new SyntheticData(dim, numClusters, vecs, vc, qs, qc, centers);
  }
}
