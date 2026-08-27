package phase0;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * Phase D latency sweep: fairness re-run of the wall-clock comparison.
 *
 * <p>Critique addressed: after Elasticsearch re-segmented the index (31 leaves,
 * largest 55.6k docs), single-entry search got cheaper for free (frontier gate
 * closes at ~151 actual dc), while the marked mechanism was still run with its
 * big-segment calibration (probe 400 + 4 rounds x 400). Fair re-evaluation:
 * sweep BOTH families' own knobs on the same fixed query set and compare the
 * resulting (wall-clock, recall) Pareto points:
 *
 * <p>Single: budget {100, 200, 400, 800, 1600}.
 * Marked-lite (probe 400, build-time entry table, 0 query-time selection):
 *   R=1/2/4 x per-beam 400; R=4 x per-beam 200; each with and without the
 *   early-stop rule (a round that contributes zero new ords to the union
 *   terminates the loop — query-time observable, no ground truth needed).
 *
 * <p>Reporting: overall + hard subset (single@100 recall &lt; 0.9) + big-leaf
 * subset, recall / mean dc / p50 / p99, so each family's Pareto frontier is
 * comparable at matched latency.
 */
public final class PhaseDLatencySweep {

  static final int K = 10;
  static final int PROBE = 400;
  static final int ROUNDS = 7;
  static final int WARMUP_PASSES = 2;

  static final class LeafState {
    int docBase;
    int size;
    float[][] vecs;
    HnswGraph graph;
    int[] nodeLevel;
    List<Integer> pool;
    int[] entryTable;
    RandomAccessVectorValues rav;
    org.apache.lucene.util.Bits accept;
  }

  static final class Query {
    int leaf;
    int qOrd;
    float[] qRaw;
    int[] ground;
    long rndSeed;
  }

  static final class Cfg {
    final String name;
    final int kind;        // 0 = single(budget); 1 = marked(rounds, perBudget, earlyStop, scoreGate)
    final int budget;
    final int rounds;
    final int perBudget;
    final boolean earlyStop;   // stop when a round adds zero new ords
    final boolean scoreGate;   // stop when a round's best score <= union's k-th score

    Cfg(String name, int budget) {
      this.name = name; this.kind = 0; this.budget = budget;
      this.rounds = 0; this.perBudget = 0; this.earlyStop = false; this.scoreGate = false;
    }

    Cfg(String name, int rounds, int perBudget, boolean earlyStop, boolean scoreGate) {
      this.name = name; this.kind = 1; this.budget = 0;
      this.rounds = rounds; this.perBudget = perBudget;
      this.earlyStop = earlyStop; this.scoreGate = scoreGate;
    }
  }

  static final Cfg[] CFGS = {
      new Cfg("single-100", 100),
      new Cfg("single-200", 200),
      new Cfg("single-400", 400),
      new Cfg("single-800", 800),
      new Cfg("single-1600", 1600),
      new Cfg("mk-R1x400", 1, 400, false, false),
      new Cfg("mk-R2x400", 2, 400, false, false),
      new Cfg("mk-R4x400", 4, 400, false, false),
      new Cfg("mk-R1x200", 1, 200, false, false),
      new Cfg("mk-R2x200", 2, 200, false, false),
      new Cfg("mk-R4x200", 4, 200, false, false),
      new Cfg("mk-R4x400-ES", 4, 400, true, false),
      new Cfg("mk-R4x200-ES", 4, 200, true, false),
      new Cfg("mk-R4x400-SG", 4, 400, false, true),
      new Cfg("mk-R4x200-SG", 4, 200, false, true),
  };

  static final class TimeResult {
    double recall;
    long dc;
    long ns;
  }

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 4) {
      System.err.println("usage: PhaseDLatencySweep <indexDir> <field> <numQueries> <tag>");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int numQueries = Integer.parseInt(args[2]);
    String tag = args[3];
    Path outDir = Paths.get(System.getProperty("out.dir", "out"));
    Files.createDirectories(outDir);
    Path gtCache = outDir.resolve("gt-" + field + ".bin");

    System.out.printf("[lat-sweep] indexDir=%s field=%s queries=%d tag=%s cfgs=%d%n",
        indexDir, field, numQueries, tag, CFGS.length);

    try (DirectoryReader ir = DirectoryReader.open(FSDirectory.open(indexDir, NoLockFactory.INSTANCE))) {
      FSDirectory dir = (FSDirectory) ir.directory();
      List<LeafReaderContext> leaves = ir.leaves();

      List<LeafState> states = new ArrayList<>();
      for (LeafReaderContext ctx : leaves) {
        CodecReader cr = (CodecReader) ctx.reader();
        FloatVectorValues fvv = cr.getFloatVectorValues(field);
        if (fvv == null) {
          continue;
        }
        int size = fvv.size();
        if (size < 100) {
          continue;
        }
        LeafState st = new LeafState();
        st.docBase = ctx.docBase;
        st.size = size;
        st.vecs = Phase0Real.loadVectors(fvv, size, fvv.dimension());
        st.graph = Phase0Real.getRealGraph(dir, cr, field);
        st.rav = RandomAccessVectorValues.fromFloats(java.util.Arrays.asList(st.vecs),
            st.vecs[0].length);

        st.nodeLevel = new int[size];
        java.util.Arrays.fill(st.nodeLevel, -1);
        st.pool = new ArrayList<>();
        for (int level = 1; level <= st.graph.numLevels() - 1; level++) {
          var it = st.graph.getNodesOnLevel(level);
          while (it.hasNext()) {
            int n = it.nextInt();
            st.nodeLevel[n] = level;
            st.pool.add(n);
          }
        }
        Random rndE = new Random(2026L * ctx.docBase + size);
        st.entryTable = new int[4];
        Set<Integer> seen = new HashSet<>();
        int ei = 0;
        while (ei < 4 && st.pool.size() > 0) {
          int n = st.pool.get(rndE.nextInt(st.pool.size()));
          if (seen.add(n)) {
            st.entryTable[ei++] = n;
          }
        }

        org.apache.lucene.util.Bits liveDocs = ctx.reader().getLiveDocs();
        RandomAccessVectorValues rav0 = fvv instanceof RandomAccessVectorValues r ? r : null;
        final RandomAccessVectorValues ravF = rav0;
        final int sizeF = size;
        st.accept = (liveDocs != null && ravF != null) ? new org.apache.lucene.util.Bits() {
          @Override
          public boolean get(int ord) {
            return liveDocs.get(ravF.ordToDoc(ord));
          }

          @Override
          public int length() {
            return sizeF;
          }
        } : null;
        states.add(st);
      }
      System.out.printf("[lat-sweep] %d leaves prepared%n", states.size());

      // ---- fixed query set (cached GT) ----
      List<Query> queries = new ArrayList<>();
      Random rnd = new Random(888L);
      if (!loadGt(gtCache, queries, numQueries, states, rnd)) {
        System.err.println("[lat-sweep] GT cache missing; run PhaseDLatency first");
        System.exit(3);
      }
      System.out.printf("[lat-sweep] ground truth loaded (%d queries)%n", queries.size());

      // ---- warmup ----
      for (int pass = 0; pass < WARMUP_PASSES; pass++) {
        for (Query qu : queries) {
          LeafState st = states.get(qu.leaf);
          InstrumentedSimSearch.ScorerFn scorer = scorerFor(st, qu.qRaw);
          for (Cfg cfg : CFGS) {
            run(cfg, st, scorer, qu, new Random(qu.rndSeed));
          }
        }
        System.out.printf("[lat-sweep] warmup pass %d/%d done%n", pass + 1, WARMUP_PASSES);
      }

      // ---- measurement ----
      int nCfg = CFGS.length;
      long[][][] times = new long[nCfg][numQueries][ROUNDS];
      double[][] rec = new double[nCfg][numQueries];
      long[][] dc = new long[nCfg][numQueries];

      for (int round = 0; round < ROUNDS; round++) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < nCfg; i++) {
          order.add(i);
        }
        Collections.shuffle(order, new Random(1000L + round));
        for (int qi = 0; qi < numQueries; qi++) {
          Query qu = queries.get(qi);
          LeafState st = states.get(qu.leaf);
          InstrumentedSimSearch.ScorerFn scorer = scorerFor(st, qu.qRaw);
          for (int c : order) {
            TimeResult tr = run(CFGS[c], st, scorer, qu, new Random(qu.rndSeed));
            times[c][qi][round] = tr.ns;
            if (round == 0) {
              rec[c][qi] = tr.recall;
              dc[c][qi] = tr.dc;
            }
          }
        }
        System.out.printf("[lat-sweep] measurement round %d/%d done%n", round + 1, ROUNDS);
      }

      // ---- per-query medians + aggregates ----
      double[][] med = new double[nCfg][numQueries];
      for (int c = 0; c < nCfg; c++) {
        for (int qi = 0; qi < numQueries; qi++) {
          long[] ts = times[c][qi];
          Arrays.sort(ts);
          med[c][qi] = ts[ROUNDS / 2];
        }
      }

      // population masks
      boolean[] hard = new boolean[numQueries];
      int hardCount = 0;
      for (int qi = 0; qi < numQueries; qi++) {
        hard[qi] = rec[0][qi] < 0.9;   // single-100 recall < 0.9
        if (hard[qi]) {
          hardCount++;
        }
      }
      int bigLeaf = 0;
      for (int i = 1; i < states.size(); i++) {
        if (states.get(i).size > states.get(bigLeaf).size) {
          bigLeaf = i;
        }
      }
      int bigCount = 0;
      for (Query qu : queries) {
        if (qu.leaf == bigLeaf) {
          bigCount++;
        }
      }
      System.out.printf("[lat-sweep] hard subset n=%d (%.1f%%), big-leaf subset n=%d%n",
          hardCount, 100.0 * hardCount / numQueries, bigCount);

      printTable("ALL queries", numQueries, null, rec, dc, med);
      printTable("HARD subset (single@100<0.9)", hardCount, hard, rec, dc, med);
      printTable("BIG-LEAF subset (size=" + states.get(bigLeaf).size + ")",
          bigCount, null, rec, dc, med, queries, bigLeaf);

      // CSV
      List<String> csv = new ArrayList<>();
      StringBuilder hdr = new StringBuilder("q,leaf,qOrd,hard");
      for (Cfg cfg : CFGS) {
        hdr.append(',').append(cfg.name).append("-rec");
      }
      for (Cfg cfg : CFGS) {
        hdr.append(',').append(cfg.name).append("-ns");
      }
      for (Cfg cfg : CFGS) {
        hdr.append(',').append(cfg.name).append("-dc");
      }
      csv.add(hdr.toString());
      for (int qi = 0; qi < numQueries; qi++) {
        StringBuilder s = new StringBuilder();
        s.append(qi).append(',').append(queries.get(qi).leaf).append(',')
            .append(queries.get(qi).qOrd).append(',').append(hard[qi]);
        for (int c = 0; c < nCfg; c++) {
          s.append(String.format(Locale.ROOT, ",%.3f", rec[c][qi]));
        }
        for (int c = 0; c < nCfg; c++) {
          s.append(',').append((long) med[c][qi]);
        }
        for (int c = 0; c < nCfg; c++) {
          s.append(',').append(dc[c][qi]);
        }
        csv.add(s.toString());
      }
      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
          outDir.resolve("phaseD-sweep-" + tag + ".csv"), StandardCharsets.UTF_8))) {
        for (String line : csv) {
          pw.println(line);
        }
      }
      System.out.printf("[lat-sweep] wrote out/phaseD-sweep-%s.csv%n", tag);
    }
  }

  static void printTable(String title, int n, boolean[] mask,
                         double[][] rec, long[][] dc, double[][] med) {
    printTable(title, n, mask, rec, dc, med, null, -1);
  }

  static void printTable(String title, int n, boolean[] mask,
                         double[][] rec, long[][] dc, double[][] med,
                         List<Query> queries, int leafFilter) {
    if (n == 0) {
      return;
    }
    System.out.printf(Locale.ROOT, "%n== %s (n=%d) ==%n", title, n);
    System.out.printf(Locale.ROOT, "%-16s %8s %9s %9s %9s %9s%n",
        "config", "recall", "dc(mean)", "p50(ms)", "p90(ms)", "p99(ms)");
    double[] lats = new double[n];
    for (int c = 0; c < CFGS.length; c++) {
      double rSum = 0, dSum = 0;
      int idx = 0;
      for (int qi = 0; qi < rec[c].length; qi++) {
        boolean in = (mask != null) ? mask[qi]
            : (queries != null) ? queries.get(qi).leaf == leafFilter : true;
        if (in) {
          rSum += rec[c][qi];
          dSum += dc[c][qi];
          lats[idx++] = med[c][qi];
        }
      }
      Arrays.sort(lats, 0, idx);
      System.out.printf(Locale.ROOT, "%-16s %8.3f %9.0f %9.3f %9.3f %9.3f%n",
          CFGS[c].name, rSum / n, dSum / n, lats[idx / 2] / 1e6,
          lats[(int) Math.ceil(0.90 * idx) - 1] / 1e6,
          lats[(int) Math.ceil(0.99 * idx) - 1] / 1e6);
    }
  }

  // ---------------- execution ----------------

  static TimeResult run(Cfg cfg, LeafState st, InstrumentedSimSearch.ScorerFn scorer,
                        Query qu, Random rnd) throws IOException {
    long t0 = System.nanoTime();
    if (cfg.kind == 0) {
      InstrumentedSimSearch.Result r = InstrumentedSimSearch.search(
          st.graph, st.size, scorer, K, cfg.budget, st.accept);
      long t1 = System.nanoTime();
      TimeResult out = new TimeResult();
      out.ns = t1 - t0;
      out.dc = r.scoreComps;
      out.recall = Phase0Real.recallAt(qu.ground, r.topK);
      return out;
    }

    // marked-lite: probe 400 + up to cfg.rounds x cfg.perBudget, early-stop optional
    InstrumentedSimSearch.Result probe = InstrumentedSimSearch.search(
        st.graph, st.size, scorer, K, PROBE, st.accept);
    long dcSum = probe.scoreComps;
    List<float[]> union = new ArrayList<>();
    Set<Integer> unionOrds = new HashSet<>();
    for (int j = 0; j < probe.topK.length; j++) {
      union.add(new float[]{probe.topK[j], probe.topKScores[j]});
      unionOrds.add(probe.topK[j]);
    }
    BitSet marked = new BitSet(st.size);
    marked.or(probe.visited);
    int[] table = st.entryTable;
    for (int step = 0; step < cfg.rounds; step++) {
      int best = -1;
      for (int i = 0; i < table.length; i++) {
        int cand = table[(step + i) % table.length];
        if (!marked.get(cand)) {
          best = cand;
          break;
        }
      }
      if (best < 0) {
        break;
      }
      InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
          st.graph, st.size, scorer, K, cfg.perBudget, st.accept, best,
          st.nodeLevel[best], marked);
      dcSum += rr.scoreComps;
      marked.or(rr.visited);
      int newOrds = 0;
      for (int j = 0; j < rr.topK.length; j++) {
        union.add(new float[]{rr.topK[j], rr.topKScores[j]});
        if (unionOrds.add(rr.topK[j])) {
          newOrds++;
        }
      }
      if (cfg.earlyStop && newOrds == 0) {
        break;
      }
      if (cfg.scoreGate) {
        float roundMax = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < rr.topKScores.length; j++) {
          roundMax = Math.max(roundMax, rr.topKScores[j]);
        }
        float[] sc = new float[union.size()];
        for (int j = 0; j < union.size(); j++) {
          sc[j] = union.get(j)[1];
        }
        Arrays.sort(sc);
        int kthIdx = Math.max(0, sc.length - K);
        if (roundMax <= sc[kthIdx]) {
          break;   // this round cannot enter the union's top-k
        }
      }
    }
    double recall = unionRecall(qu.ground, union, K);
    long t1 = System.nanoTime();
    TimeResult out = new TimeResult();
    out.ns = t1 - t0;
    out.dc = dcSum;
    out.recall = recall;
    return out;
  }

  static double unionRecall(int[] ground, List<float[]> union, int k) {
    List<float[]> sorted = new ArrayList<>(union);
    sorted.sort((a, b) -> Float.compare(b[1], a[1]));
    Set<Integer> seen = new HashSet<>();
    int[] top = new int[Math.min(k, sorted.size())];
    int idx = 0;
    for (float[] e : sorted) {
      int ord = (int) e[0];
      if (seen.add(ord)) {
        top[idx++] = ord;
        if (idx == top.length) {
          break;
        }
      }
    }
    return Phase0Real.recallAt(ground, top);
  }

  // ---------------- shared helpers ----------------

  static InstrumentedSimSearch.ScorerFn scorerFor(LeafState st, float[] qRaw) throws IOException {
    RandomVectorScorer rs = DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
        VectorSimilarityFunction.COSINE, st.rav, qRaw);
    return ord -> {
      try {
        return rs.score(ord);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    };
  }

  static boolean loadGt(Path cache, List<Query> queries, int numQueries,
                        List<LeafState> states, Random rnd) throws IOException {
    if (!Files.exists(cache)) {
      return false;
    }
    try (DataInputStream in = new DataInputStream(Files.newInputStream(cache))) {
      if (in.readInt() != 0x6A7A || in.readInt() != numQueries) {
        return false;
      }
      int nLeaves = in.readInt();
      if (nLeaves != states.size()) {
        return false;
      }
      for (int i = 0; i < nLeaves; i++) {
        if (in.readInt() != states.get(i).size) {
          return false;
        }
      }
      for (int q = 0; q < numQueries; q++) {
        Query qu = new Query();
        qu.leaf = in.readInt();
        qu.qOrd = in.readInt();
        LeafState st = states.get(qu.leaf);
        qu.qRaw = st.vecs[qu.qOrd];
        qu.ground = new int[K];
        for (int i = 0; i < K; i++) {
          qu.ground[i] = in.readInt();
        }
        qu.rndSeed = 0x5EEDL ^ ((long) st.docBase * 131071L + qu.qOrd);
        queries.add(qu);
        rnd.nextInt(states.size());
        rnd.nextInt(st.size);
      }
      return true;
    }
  }
}
