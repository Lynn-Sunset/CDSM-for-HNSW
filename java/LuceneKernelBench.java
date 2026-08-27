// Lucene 9.12 内核单价微基准：VectorSimilarityFunction.DOT_PRODUCT（内部走 Panama
// Vector API，若 JVM 支持）vs Phase0Real.dot 标量循环，在 128/200/960 维各测 10M 次。
// 目的：把「Java 端口 vs faiss」的墙钟差距分解为 (内核实现) 与 (搜索结构) 两轴——
// 真实 Lucene 的 per-dc 单价应落在标量端口与 faiss SIMD 之间。
// 需 --add-modules jdk.incubator.vector（JDK21）；不支持时打印 provider 类名以披露。
package phase0;

import java.util.Arrays;
import java.util.Random;
import org.apache.lucene.index.VectorSimilarityFunction;

public class LuceneKernelBench {
  public static void main(String[] args) {
    String provider = "unknown";
    try {
      provider = org.apache.lucene.internal.vectorization.VectorizationProvider
          .getInstance().getClass().getName();
    } catch (Throwable t) {
      provider = "unavailable: " + t;
    }
    System.out.println("[kernel-bench] VectorizationProvider = " + provider);

    int[] dims = {128, 200, 960};
    int iters = 20_000_000;
    Random rnd = new Random(42);
    float sink = 0f;
    for (int dim : dims) {
      float[] a = new float[dim];
      float[] b = new float[dim];
      for (int i = 0; i < dim; i++) {
        a[i] = rnd.nextFloat();
        b[i] = rnd.nextFloat();
      }
      // 预热
      for (int i = 0; i < 200_000; i++) {
        sink += VectorSimilarityFunction.DOT_PRODUCT.compare(a, b);
        sink += Phase0Real.dot(a, b);
      }
      // lucene (Panama or scalar fallback)
      long t0 = System.nanoTime();
      for (int i = 0; i < iters; i++) {
        sink += VectorSimilarityFunction.DOT_PRODUCT.compare(a, b);
      }
      long t1 = System.nanoTime();
      double luceneNs = (t1 - t0) / (double) iters;
      // 标量端口
      long t2 = System.nanoTime();
      for (int i = 0; i < iters; i++) {
        sink += Phase0Real.dot(a, b);
      }
      long t3 = System.nanoTime();
      double scalarNs = (t3 - t2) / (double) iters;
      System.out.printf("[kernel-bench] dim=%4d  lucene=%.1f ns/dot  scalar-port=%.1f ns/dot  ratio=%.2fx%n",
          dim, luceneNs, scalarNs, scalarNs / luceneNs);
    }
    System.out.println("[kernel-bench] sink=" + (int) sink + " (anti-DCE)");
  }
}
