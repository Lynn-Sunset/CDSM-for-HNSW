package phase0;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.hnsw.HnswGraph;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * Real-data visualization extractor for the theory figures: finds REAL
 * catastrophic SIFT queries (single@100 = 0/10) that CDSM-Lite-R4 fully
 * rescues (recall >= 0.8), and exports:
 *   realvis-q{idx}.csv          — vectors of q / gt / ctx / single / r1..r4 / entries / desc
 *   realvis-q{idx}-nodes.csv    — graph node set S = visited ∪ GT ∪ N(GT) ∪ entries ∪ desc:
 *                                 tag,ord,score (score = -dist2(q, base[ord]))
 *   realvis-q{idx}-edges.csv    — level-0 edges (u,v) within S
 *   realvis-q{idx}-meta.txt     — per-query recall/coverage stats + probe gate theta
 * For the threshold-subgraph (G_theta) analysis and graph-layout figures.
 *
 * <p>Usage: PhaseRealVis &lt;dataDir&gt;
 */
public final class PhaseRealVis {

  static final int K = 10;
  static final int PROBE = 400;
  static final int BUDGET = 400;
  static final int R = 4;

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 1) {
      System.err.println("usage: PhaseRealVis <dataDir>");
      System.exit(2);
    }
    Path dir = Paths.get(args[0]);
    Path outDir = Paths.get(System.getProperty("out.dir", "out"));

    System.out.println("[realvis] loading base/queries/gt ...");
    float[][] base = PhaseSIFT.loadFvecs(dir.resolve("sift_base.fvecs"));
    float[][] queries = PhaseSIFT.loadFvecs(dir.resolve("sift_query.fvecs"));
    int[][] gt = PhaseSIFT.loadIvecs(dir.resolve("sift_groundtruth.ivecs"), 10);
    int n = base.length;

    System.out.println("[realvis] loading cached graph ...");
    Path graphFile = outDir.resolve("sift-hnsw-1M.hnsw");
    if (!Files.exists(graphFile)) {
      throw new IllegalStateException("cached graph missing: " + graphFile);
    }
    HnswGraph graph = PhaseHnswStore.load(graphFile);

    int[] nodeLevel = new int[n];
    java.util.Arrays.fill(nodeLevel, -1);
    List<Integer> pool = new ArrayList<>();
    for (int level = 1; level <= graph.numLevels() - 1; level++) {
      var it = graph.getNodesOnLevel(level);
      while (it.hasNext()) {
        int node = it.nextInt();
        nodeLevel[node] = level;
        pool.add(node);
      }
    }
    int[] entryTable = new int[4];
    Random rndE = new Random(2026);
    Set<Integer> seen = new HashSet<>();
    int ei = 0;
    while (ei < 4) {
      int cand = pool.get(rndE.nextInt(pool.size()));
      if (seen.add(cand)) {
        entryTable[ei++] = cand;
      }
    }
    System.out.printf("[realvis] pool=%d entries=%s%n", pool.size(),
        java.util.Arrays.toString(entryTable));

    // ---- scan the Random(777)-sampled stream for catastrophic + fully-rescued ----
    Random rndQ = new Random(777);
    List<Integer> targets = new ArrayList<>();
    List<float[]> tq = new ArrayList<>();
    List<int[]> tgt = new ArrayList<>();
    for (int qi = 0; qi < 2000 && targets.size() < 2; qi++) {
      int qIdx = rndQ.nextInt(queries.length);
      final float[] qv = queries[qIdx];
      final float[][] baseF = base;
      InstrumentedSimSearch.ScorerFn scorer = ord -> -InstrumentedSearch.dist2(qv, baseF[ord]);
      InstrumentedSimSearch.Result r100 = InstrumentedSimSearch.search(
          graph, n, scorer, K, 100, null);
      if (Phase0Real.recallAt(gt[qIdx], r100.topK) > 0) {
        continue;
      }
      List<float[]> union = new ArrayList<>();
      InstrumentedSimSearch.Result probe = InstrumentedSimSearch.search(
          graph, n, scorer, K, PROBE, null);
      for (int j = 0; j < probe.topK.length; j++) {
        union.add(new float[]{probe.topK[j], probe.topKScores[j]});
      }
      BitSet marked = new BitSet(n);
      marked.or(probe.visited);
      for (int step = 0; step < R; step++) {
        int best = -1;
        for (int i = 0; i < entryTable.length; i++) {
          int cand = entryTable[(step + i) % entryTable.length];
          if (!marked.get(cand)) {
            best = cand;
            break;
          }
        }
        if (best < 0) {
          break;
        }
        InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
            graph, n, scorer, K, BUDGET, null, best, nodeLevel[best], marked);
        marked.or(rr.visited);
        for (int j = 0; j < rr.topK.length; j++) {
          union.add(new float[]{rr.topK[j], rr.topKScores[j]});
        }
      }
      union.sort((a, b) -> Float.compare(b[1], a[1]));
      Set<Integer> topSet = new HashSet<>();
      int[] top = new int[Math.min(K, union.size())];
      int ti = 0;
      for (float[] e : union) {
        if (topSet.add((int) e[0])) {
          top[ti++] = (int) e[0];
          if (ti == top.length) {
            break;
          }
        }
      }
      double rec = Phase0Real.recallAt(gt[qIdx], top);
      if (rec >= 0.8) {
        targets.add(qIdx);
        tq.add(qv);
        tgt.add(gt[qIdx]);
        System.out.printf("[realvis] candidate: query idx=%d (sampled #%d), "
                + "single100=0/10, CDSM-Lite-R4 recall=%.1f%n", qIdx, qi, rec);
      }
    }
    if (targets.isEmpty()) {
      throw new IllegalStateException("no qualifying catastrophic query found");
    }

    // ---- export each target ----
    for (int t = 0; t < targets.size(); t++) {
      int qIdx = targets.get(t);
      final float[] qv = tq.get(t);
      final int[] ground = tgt.get(t);
      final float[][] baseF = base;
      InstrumentedSimSearch.ScorerFn scorer = ord -> -InstrumentedSearch.dist2(qv, baseF[ord]);

      InstrumentedSimSearch.Result probe = InstrumentedSimSearch.search(
          graph, n, scorer, K, PROBE, null);
      BitSet marked = new BitSet(n);
      marked.or(probe.visited);
      List<BitSet> roundNew = new ArrayList<>();
      for (int step = 0; step < R; step++) {
        int best = -1;
        for (int i = 0; i < entryTable.length; i++) {
          int cand = entryTable[(step + i) % entryTable.length];
          if (!marked.get(cand)) {
            best = cand;
            break;
          }
        }
        if (best < 0) {
          break;
        }
        BitSet before = (BitSet) marked.clone();
        InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
            graph, n, scorer, K, BUDGET, null, best, nodeLevel[best], marked);
        marked.or(rr.visited);
        BitSet fresh = (BitSet) rr.visited.clone();
        fresh.andNot(before);
        roundNew.add(fresh);
      }

      Path out = outDir.resolve("realvis-q" + qIdx + ".csv");
      StringBuilder meta = new StringBuilder();
      meta.append("queryIdx=").append(qIdx).append('\n');
      meta.append("single100rec=0\n");
      meta.append("single400dc=").append(probe.scoreComps).append('\n');
      meta.append("singleVisited=").append(probe.visited.cardinality()).append('\n');
      for (int s = 0; s < roundNew.size(); s++) {
        meta.append("round").append(s + 1).append("New=")
            .append(roundNew.get(s).cardinality()).append('\n');
      }
      int gtInSingle = 0, gtInRounds = 0;
      for (int g : ground) {
        if (probe.visited.get(g)) {
          gtInSingle++;
        }
        for (BitSet rs : roundNew) {
          if (rs.get(g)) {
            gtInRounds++;
            break;
          }
        }
      }
      meta.append("gtInSingle=").append(gtInSingle).append('\n');
      meta.append("gtInRoundsOnly=").append(gtInRounds).append('\n');
      float theta = Float.POSITIVE_INFINITY;
      for (float s2 : probe.topKScores) {
        theta = Math.min(theta, s2);
      }
      meta.append("probeTheta=").append(theta).append('\n');
      meta.append("probeBest=").append(probe.topKScores[0]).append('\n');
      Files.write(outDir.resolve("realvis-q" + qIdx + "-meta.txt"),
          meta.toString().getBytes(StandardCharsets.UTF_8));
      System.out.printf("[realvis] %s%n", meta);

      Random ctxRnd = new Random(1234 + qIdx);
      Set<Integer> ctx = new HashSet<>();
      while (ctx.size() < 500) {
        ctx.add(ctxRnd.nextInt(n));
      }
      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
        StringBuilder hdr = new StringBuilder("tag,ord");
        for (int d = 0; d < base[0].length; d++) {
          hdr.append(",v").append(d);
        }
        pw.println(hdr);
        writeRow(pw, "q", 0, qv);
        for (int g : ground) {
          writeRow(pw, "gt", g, base[g]);
        }
        for (int c : ctx) {
          writeRow(pw, "ctx", c, base[c]);
        }
        for (int i = probe.visited.nextSetBit(0); i >= 0; i = probe.visited.nextSetBit(i + 1)) {
          writeRow(pw, "single", i, base[i]);
        }
        for (int s = 0; s < roundNew.size(); s++) {
          BitSet rs = roundNew.get(s);
          for (int i = rs.nextSetBit(0); i >= 0; i = rs.nextSetBit(i + 1)) {
            writeRow(pw, "r" + (s + 1), i, base[i]);
          }
        }
        for (int i = 0; i < entryTable.length; i++) {
          writeRow(pw, "entry" + (i + 1), entryTable[i], base[entryTable[i]]);
        }
        writeRow(pw, "desc", probe.descendedNode, base[probe.descendedNode]);
      }
      System.out.printf("[realvis] wrote %s%n", out);
      exportGraph(probe, roundNew, ground, entryTable, base, qv, graph, outDir, qIdx);
    }
  }

  /** Export the induced level-0 subgraph on S = visited merged with GT merged with
   *  N(GT) merged with entries merged with desc: nodes file (tag,ord,score) + edges
   *  file (u,v), for the threshold-subgraph (G_theta) analysis and graph figures. */
  static void exportGraph(InstrumentedSimSearch.Result probe, List<BitSet> roundNew,
                          int[] ground, int[] entryTable, float[][] base, float[] qv,
                          HnswGraph graph, Path outDir, int qIdx) throws IOException {
    Set<Integer> s = new java.util.TreeSet<>();
    for (int i = probe.visited.nextSetBit(0); i >= 0; i = probe.visited.nextSetBit(i + 1)) {
      s.add(i);
    }
    for (BitSet rs : roundNew) {
      for (int i = rs.nextSetBit(0); i >= 0; i = rs.nextSetBit(i + 1)) {
        s.add(i);
      }
    }
    for (int g : ground) {
      s.add(g);
    }
    for (int e : entryTable) {
      s.add(e);
    }
    s.add(probe.descendedNode);
    Set<Integer> gtnb = new HashSet<>();
    for (int g : ground) {
      graph.seek(0, g);
      int x;
      while ((x = graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
        if (!s.contains(x)) {
          s.add(x);
          gtnb.add(x);
        }
      }
    }
    try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
        outDir.resolve("realvis-q" + qIdx + "-nodes.csv"), StandardCharsets.UTF_8))) {
      pw.println("tag,ord,score");
      for (int g : ground) {
        pw.println("gt," + g + "," + -InstrumentedSearch.dist2(qv, base[g]));
      }
      for (int nb : gtnb) {
        pw.println("gtnb," + nb + "," + -InstrumentedSearch.dist2(qv, base[nb]));
      }
      for (int i = probe.visited.nextSetBit(0); i >= 0; i = probe.visited.nextSetBit(i + 1)) {
        pw.println("single," + i + "," + -InstrumentedSearch.dist2(qv, base[i]));
      }
      for (int si = 0; si < roundNew.size(); si++) {
        BitSet rs = roundNew.get(si);
        for (int i = rs.nextSetBit(0); i >= 0; i = rs.nextSetBit(i + 1)) {
          pw.println("r" + (si + 1) + "," + i + "," + -InstrumentedSearch.dist2(qv, base[i]));
        }
      }
      for (int i = 0; i < entryTable.length; i++) {
        pw.println("entry" + (i + 1) + "," + entryTable[i] + ","
            + -InstrumentedSearch.dist2(qv, base[entryTable[i]]));
      }
      pw.println("desc," + probe.descendedNode + ","
          + -InstrumentedSearch.dist2(qv, base[probe.descendedNode]));
    }
    Set<String> edges = new HashSet<>();
    int[] arr = new int[s.size()];
    int ai = 0;
    for (int u : s) {
      arr[ai++] = u;
    }
    java.util.Arrays.sort(arr);
    Set<Integer> nodeSet = new HashSet<>(s);
    for (int u : arr) {
      graph.seek(0, u);
      int x;
      while ((x = graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
        if (x == u || !nodeSet.contains(x)) {
          continue;
        }
        int a = Math.min(u, x), b = Math.max(u, x);
        edges.add(a + "," + b);
      }
    }
    try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
        outDir.resolve("realvis-q" + qIdx + "-edges.csv"), StandardCharsets.UTF_8))) {
      pw.println("u,v");
      for (String e : edges) {
        pw.println(e);
      }
    }
    System.out.printf("[realvis] exported graph: S=%d nodes, %d edges%n", s.size(), edges.size());
  }

  static void writeRow(PrintWriter pw, String tag, int ord, float[] vec) {
    StringBuilder sb = new StringBuilder();
    sb.append(tag).append(',').append(ord);
    for (float v : vec) {
      sb.append(String.format(Locale.ROOT, ",%.6f", v));
    }
    pw.println(sb);
  }
}
