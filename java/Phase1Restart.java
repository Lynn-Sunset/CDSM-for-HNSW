package phase0;

import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.NoLockFactory;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * Phase 1 (RQ3 revised): restart from ALL upper-layer nodes (level >= 1),
 * not just the top level.
 *
 * <p>Per query: primary search -> ground truth -> classification -> oracle:
 * R alternative entries sampled from the upper-layer pool by "farthest from
 * the descended node" diversity, each re-searched with descent starting at
 * its own level (full budget). Records the cumulative best-recall curve over
 * R (the ceiling of the mechanism) and the Lucene-native multi-seed beam
 * variant (same budget, shared visited). Also records improvement traces of
 * the primary search for trigger-signal analysis (revised RQ1).
 *
 * <p>Usage: Phase1Restart &lt;indexDir&gt; &lt;field&gt; &lt;numQueries&gt; &lt;visitLimit&gt;
 * &lt;tag&gt; [R] [sampleCandidates]
 */
public final class Phase1Restart {

  static final int K = 10;

  record Pair(int ord, float d) {}

  public static void main(String[] args) throws IOException {
    EsLoggingBootstrap.install();
    if (args.length < 5) {
      System.err.println("usage: Phase1Restart <indexDir> <field> <numQueries> <visitLimit> <tag> [R] [sampleCandidates]");
      System.exit(2);
    }
    Path indexDir = Paths.get(args[0]);
    String field = args[1];
    int numQueries = Integer.parseInt(args[2]);
    int visitLimit = Integer.parseInt(args[3]);
    String tag = args[4];
    int R = args.length >= 6 ? Integer.parseInt(args[5]) : 16;
    int sampleCandidates = args.length >= 7 ? Integer.parseInt(args[6]) : 500;

    System.out.printf("[phase1] indexDir=%s field=%s queries=%d visitLimit=%d tag=%s R=%d sample=%d%n",
        indexDir, field, numQueries, visitLimit, tag, R, sampleCandidates);

    try (DirectoryReader ir = DirectoryReader.open(FSDirectory.open(indexDir, NoLockFactory.INSTANCE))) {
      FSDirectory dir = (FSDirectory) ir.directory();
      List<LeafReaderContext> leaves = ir.leaves();
      Agg agg = new Agg();

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
        float[][] vecs = Phase0Real.loadVectors(fvv, size, fvv.dimension());
        HnswGraph graph = Phase0Real.getRealGraph(dir, cr, field);

        // upper-layer pool: union of levels 1..L-1
        int[] nodeLevel = new int[size];
        java.util.Arrays.fill(nodeLevel, -1);
        List<Integer> pool = new ArrayList<>();
        for (int level = 1; level <= graph.numLevels() - 1; level++) {
          var it = graph.getNodesOnLevel(level);
          while (it.hasNext()) {
            int n = it.nextInt();
            nodeLevel[n] = level;
            pool.add(n);
          }
        }
        System.out.printf("[leaf docBase=%d] size=%d levels=%d poolUpper=%d (%.1f%%)%n",
            ctx.docBase, size, graph.numLevels(), pool.size(), 100.0 * pool.size() / size);

        // live-docs accept bits
        org.apache.lucene.util.Bits liveDocs = ctx.reader().getLiveDocs();
        RandomAccessVectorValues rav = fvv instanceof RandomAccessVectorValues r ? r : null;
        final RandomAccessVectorValues ravF = rav;
        final int sizeF = size;
        org.apache.lucene.util.Bits accept = (liveDocs != null && ravF != null)
            ? new org.apache.lucene.util.Bits() {
              @Override
              public boolean get(int ord) {
                return liveDocs.get(ravF.ordToDoc(ord));
              }

              @Override
              public int length() {
                return sizeF;
              }
            } : null;

        int qPerLeaf = Math.max(1, numQueries / leaves.size());
        Random rnd = new Random(1234L * ctx.docBase + size);
        for (int q = 0; q < qPerLeaf; q++) {
          int qOrd = rnd.nextInt(size);
          float[] qNorm = Phase0Real.normalize(vecs[qOrd]);
          InstrumentedSimSearch.ScorerFn scorer = ord -> Phase0Real.dot(qNorm, vecs[ord]);

          // primary search (traced)
          InstrumentedSimSearch.Result r1 =
              InstrumentedSimSearch.search(graph, size, scorer, K, visitLimit, accept);
          int[] ground = Phase0Real.bruteForceTopK(vecs, qNorm, K, accept);
          // idea-3 variant: beam seeded with descent's 2nd/3rd best scored nodes
          // (free seeds, same budget)
          InstrumentedSimSearch.Result rlm = InstrumentedSimSearch.searchLocalMultiSeed(
              graph, size, scorer, K, visitLimit, accept, 2);
          double lmRec = Phase0Real.recallAt(ground, rlm.topK);
          long lmD = rlm.scoreComps;
          double baseRec = Phase0Real.recallAt(ground, r1.topK);
          String cls = baseRec >= 0.9 ? "GOOD" : baseRec <= 0.3 ? "BAD_BASIN" : "PARTIAL";

          // trigger signals from the primary trace
          int impCount = r1.improveTrace.length;
          int lateImps = 0;
          for (int step : r1.improveTrace) {
            if (step > 0.8 * r1.scoreComps) {
              lateImps++;
            }
          }
          double lateFrac = impCount == 0 ? 0 : (double) lateImps / impCount;

          double ceilingRec = baseRec;
          double bestAt1 = baseRec, bestAt2 = baseRec, bestAt4 = baseRec,
              bestAt8 = baseRec, bestAt16 = baseRec;
          double multiRec = baseRec;
          long restartCost = 0;
          long multiD = r1.scoreComps;

          // decisive control: single searches at larger budgets (same-cost comparison)
          double rec200 = -1, rec400 = -1, rec800 = -1, rec1600 = -1;
          double multi1600 = -1;

          if (!cls.equals("GOOD") && pool.size() >= 2) {
            rec200 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
                graph, size, scorer, K, 200, accept).topK);
            rec400 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
                graph, size, scorer, K, 400, accept).topK);
            rec800 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
                graph, size, scorer, K, 800, accept).topK);
            rec1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.search(
                graph, size, scorer, K, 1600, accept).topK);
            // sample candidates, keep the R farthest from the descended node
            List<Integer> alts = selectFarthest(pool, r1.descendedNode, vecs,
                Math.min(sampleCandidates, pool.size()), Math.min(R, pool.size()), rnd);

            double runningBest = baseRec;
            for (int i = 0; i < alts.size(); i++) {
              int alt = alts.get(i);
              InstrumentedSimSearch.Result rr = InstrumentedSimSearch.searchFrom(
                  graph, size, scorer, K, visitLimit, accept, alt, nodeLevel[alt]);
              restartCost += rr.scoreComps;
              double rc = Phase0Real.recallAt(ground, rr.topK);
              if (rc > runningBest) {
                runningBest = rc;
              }
              if (i == 0) bestAt1 = runningBest;
              else if (i == 1) bestAt2 = runningBest;
              else if (i == 3) bestAt4 = runningBest;
              else if (i == 7) bestAt8 = runningBest;
              else if (i == 15) bestAt16 = runningBest;
            }
            ceilingRec = runningBest;

            // multi-seed beam (shared budget, shared visited)
            int[] extras = new int[alts.size()];
            for (int i = 0; i < alts.size(); i++) {
              extras[i] = alts.get(i);
            }
            InstrumentedSimSearch.Result rm = InstrumentedSimSearch.searchMultiSeed(
                graph, size, scorer, K, visitLimit, accept, graph.entryNode(), extras);
            multiD = rm.scoreComps;
            multiRec = Phase0Real.recallAt(ground, rm.topK);

            // multi-seed with a large shared budget (fair vs 16 restarts)
            multi1600 = Phase0Real.recallAt(ground, InstrumentedSimSearch.searchMultiSeed(
                graph, size, scorer, K, 1600, accept, graph.entryNode(), extras).topK);
          }

          // ---- user proposal: sequential restarts with path marking ----
          // Each restart skips all previously visited nodes (never re-scored):
          // it only pays for NEW territory. Entry = farthest UNVISITED
          // upper-layer node from the primary descended node.
          double seqAt1 = baseRec, seqAt2 = baseRec, seqAt4 = baseRec,
              seqAt8 = baseRec, seqAt16 = baseRec;
          long seqCost1 = r1.scoreComps, seqCost16 = r1.scoreComps;
          if (!cls.equals("GOOD") && pool.size() >= 2) {
            java.util.BitSet marked = new java.util.BitSet(size);
            marked.or(r1.visited);
            List<float[]> union = new ArrayList<>();
            for (int i = 0; i < r1.topK.length; i++) {
              union.add(new float[]{r1.topK[i], r1.topKScores[i]});
            }
            double seqBest = baseRec;
            long seqCost = r1.scoreComps;
            final int incBudget = 100;
            for (int step = 1; step <= 16; step++) {
              int alt = farthestUnvisited(pool, marked, r1.descendedNode, vecs, 500, rnd);
              if (alt < 0) {
                break;
              }
              InstrumentedSimSearch.Result rs = InstrumentedSimSearch.searchFrom(
                  graph, size, scorer, K, incBudget, accept, alt, nodeLevel[alt], marked);
              marked.or(rs.visited);
              seqCost += rs.scoreComps;
              for (int i = 0; i < rs.topK.length; i++) {
                union.add(new float[]{rs.topK[i], rs.topKScores[i]});
              }
              seqBest = unionRecall(ground, union, K);
              if (step == 1) {
                seqAt1 = seqBest;
                seqCost1 = seqCost;
              } else if (step == 2) {
                seqAt2 = seqBest;
              } else if (step == 4) {
                seqAt4 = seqBest;
              } else if (step == 8) {
                seqAt8 = seqBest;
              } else if (step == 16) {
                seqAt16 = seqBest;
                seqCost16 = seqCost;
              }
              if (rs.scoreComps < 5) {
                break;   // no new territory left
              }
            }
          }

          agg.add(ctx.docBase * 10000 + q, size, baseRec, cls, ceilingRec,
              ceilingRec - baseRec, multiRec, multiRec - baseRec,
              bestAt1, bestAt2, bestAt4, bestAt8, bestAt16,
              r1.scoreComps, restartCost, multiD, impCount, r1.patienceRatio(),
              lateFrac, pool.size(), rec200, rec400, rec800, rec1600, multi1600,
              seqAt1, seqCost1, seqAt16, seqCost16, lmRec, lmD);
        }
      }

      System.out.printf("%n==== Phase1 restart report (tag=%s field=%s) ====%n", tag, field);
      agg.report(System.out);

      Path outDir = Paths.get(System.getProperty("out.dir", "out"));
      Files.createDirectories(outDir);
      try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
          outDir.resolve("phase1-" + tag + ".csv"), StandardCharsets.UTF_8))) {
        pw.println("qid,size,baseRec,class,ceilingRec,gain,multiRec,multiGain,bestAtR1,bestAtR2,bestAtR4,bestAtR8,bestAtR16,baseD,restartCost,multiD,impCount,patienceRatio,lateFrac,poolSize,rec200,rec400,rec800,rec1600,multi1600,seqRec1,seqCost1,seqRec16,seqCost16,lmRec,lmD");
        for (String line : agg.csvLines) {
          pw.println(line);
        }
      }
      System.out.printf("[phase1] wrote out/phase1-%s.csv%n", tag);
    }
  }

  /** Sample {@code sample} distinct pool nodes, keep the {@code r} farthest
   *  (vector-space L2) from {@code from}. */
  static List<Integer> selectFarthest(List<Integer> pool, int from, float[][] vecs,
                                      int sample, int r, Random rnd) {
    Set<Integer> picked = new HashSet<>();
    int attempts = 0;
    while (picked.size() < Math.min(sample, pool.size()) && attempts < sample * 4) {
      picked.add(pool.get(rnd.nextInt(pool.size())));
      attempts++;
    }
    record Pair(int ord, float d) {}
    List<Pair> cands = new ArrayList<>();
    for (int n : picked) {
      cands.add(new Pair(n, l2(vecs[n], vecs[from])));
    }
    cands.sort((a, b) -> Float.compare(b.d(), a.d()));
    List<Integer> out = new ArrayList<>();
    for (int i = 0; i < Math.min(r, cands.size()); i++) {
      out.add(cands.get(i).ord());
    }
    return out;
  }

  static float l2(float[] a, float[] b) {
    float s = 0f;
    for (int i = 0; i < a.length; i++) {
      float d = a[i] - b[i];
      s += d * d;
    }
    return s;
  }

  /** Sample up to {@code sample} UNVISITED pool nodes, return the one farthest
   *  from {@code from}; -1 if the pool is exhausted. */
  static int farthestUnvisited(List<Integer> pool, java.util.BitSet marked, int from,
                               float[][] vecs, int sample, Random rnd) {
    int best = -1;
    float bestD = -1f;
    int attempts = 0, found = 0;
    while (found < sample && attempts < sample * 8) {
      attempts++;
      int n = pool.get(rnd.nextInt(pool.size()));
      if (marked.get(n)) {
        continue;
      }
      found++;
      float d = l2(vecs[n], vecs[from]);
      if (d > bestD) {
        bestD = d;
        best = n;
      }
    }
    return best;
  }

  /** Recall of the top-k union (by score) of all collected results. */
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

  // ---------- aggregation ----------

  static final class Agg {
    final List<String> csvLines = new ArrayList<>();
    final List<Double> baseRecs = new ArrayList<>();
    final List<Long> ds = new ArrayList<>();
    final List<Double> ceilingRecs = new ArrayList<>();
    final List<Double> gains = new ArrayList<>();
    final List<Double> multiRecs = new ArrayList<>();
    final List<Double> multiGains = new ArrayList<>();
    final List<String> classes = new ArrayList<>();
    int good = 0, bad = 0;
    // gain buckets for signal analysis
    final List<double[]> signalRows = new ArrayList<>(); // [lateFrac, impCount, patienceRatio, gain]
    final List<Double> best1 = new ArrayList<>();
    final List<Double> best2 = new ArrayList<>();
    final List<Double> best4 = new ArrayList<>();
    final List<Double> best8 = new ArrayList<>();
    final List<Double> best16 = new ArrayList<>();
    final List<Double> restCosts = new ArrayList<>();
    final List<Double> r200 = new ArrayList<>();
    final List<Double> r400 = new ArrayList<>();
    final List<Double> r800 = new ArrayList<>();
    final List<Double> r1600 = new ArrayList<>();
    final List<Double> m1600 = new ArrayList<>();
    final List<Double> seq1 = new ArrayList<>();
    final List<Double> seqCost1s = new ArrayList<>();
    final List<Double> seq16 = new ArrayList<>();
    final List<Double> seqCost16s = new ArrayList<>();
    final List<Double> lmRecs = new ArrayList<>();
    final List<Double> lmDs = new ArrayList<>();

    void add(int qid, int size, double baseRec, String cls, double ceilingRec,
             double gain, double multiRec, double multiGain,
             double bestAt1, double bestAt2, double bestAt4, double bestAt8,
             double bestAt16, long baseD, long restartCost, long multiD,
             int impCount, double patienceRatio, double lateFrac, int poolSize,
             double rec200, double rec400, double rec800, double rec1600,
             double multi1600, double seqAt1, long seqCost1, double seqAt16,
             long seqCost16, double lmRec, long lmD) {
      baseRecs.add(baseRec);
      ds.add(baseD);
      ceilingRecs.add(ceilingRec);
      gains.add(gain);
      multiRecs.add(multiRec);
      multiGains.add(multiGain);
      classes.add(cls);
      if (cls.equals("GOOD")) good++;
      if (cls.equals("BAD_BASIN")) bad++;
      if (!cls.equals("GOOD")) {
        signalRows.add(new double[]{lateFrac, impCount, patienceRatio, gain});
        best1.add(bestAt1);
        best2.add(bestAt2);
        best4.add(bestAt4);
        best8.add(bestAt8);
        best16.add(bestAt16);
        restCosts.add((double) restartCost);
        r200.add(rec200);
        r400.add(rec400);
        r800.add(rec800);
        r1600.add(rec1600);
        m1600.add(multi1600);
        seq1.add(seqAt1);
        seqCost1s.add((double) seqCost1);
        seq16.add(seqAt16);
        seqCost16s.add((double) seqCost16);
        lmRecs.add(lmRec);
        lmDs.add((double) lmD);
      }
      csvLines.add(String.format(Locale.ROOT,
          "%d,%d,%.3f,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.3f,%d,%.3f,%d",
          qid, size, baseRec, cls, ceilingRec, gain, multiRec, multiGain,
          bestAt1, bestAt2, bestAt4, bestAt8, bestAt16,
          baseD, restartCost, multiD, impCount, patienceRatio, lateFrac, poolSize,
          rec200, rec400, rec800, rec1600, multi1600,
          seqAt1, seqCost1, seqAt16, seqCost16, lmRec, lmD));
    }

    void report(Appendable out) throws IOException {
      int n = baseRecs.size();
      int nonGood = 0;
      for (String c : classes) {
        if (!c.equals("GOOD")) nonGood++;
      }
      out.append(String.format(Locale.ROOT,
          "%n[overall] queries=%d good=%d (%.1f%%) bad_basin=%d (%.1f%%) non-good=%d%n",
          n, good, 100.0 * good / n, bad, 100.0 * bad / n, nonGood));
      out.append(String.format(Locale.ROOT,
          "  mean base recall=%.3f%n", mean(baseRecs)));

      if (nonGood > 0) {
        int gain02 = 0, gain01 = 0, any = 0;
        double gSum = 0;
        for (double g : gains) {
          if (g > 0) any++;
          if (g >= 0.1) gain01++;
          if (g >= 0.2) gain02++;
          gSum += g;
        }
        out.append(String.format(Locale.ROOT,
            "%n[oracle ceiling: R farthest upper-layer re-searches, full budget each]%n"
                + "  queries with gain>0:   %d (%.1f%%)%n"
                + "  queries with gain>=0.1: %d (%.1f%%)%n"
                + "  queries with gain>=0.2: %d (%.1f%%)%n"
                + "  mean gain=%.3f  mean ceiling recall=%.3f%n",
            any, 100.0 * any / nonGood, gain01, 100.0 * gain01 / nonGood,
            gain02, 100.0 * gain02 / nonGood, gSum / nonGood, mean(ceilingRecs)));
        out.append(String.format(Locale.ROOT,
            "  mean cumulative-best recall: base=%.3f R1=%.3f R2=%.3f R4=%.3f R8=%.3f R16=%.3f%n",
            mean(baseRecs), mean(best1), mean(best2), mean(best4), mean(best8), mean(best16)));
        out.append(String.format(Locale.ROOT,
            "  mean restart cost (16 re-searches)=%.0f dc%n", mean(restCosts)));
        out.append(String.format(Locale.ROOT,
            "[multi-seed beam, same budget] mean recall=%.3f mean gain=%.3f%n",
            mean(multiRecs), mean(multiGains)));
        out.append(String.format(Locale.ROOT,
            "%n[decisive control: single search at larger budgets vs restart ceiling]%n"
                + "  single@200 =%.3f  single@400 =%.3f  single@800 =%.3f  single@1600=%.3f%n"
                + "  multi-seed@1600 =%.3f%n"
                + "  restart ceiling (mean cost %.0f dc) =%.3f%n",
            mean(r200), mean(r400), mean(r800), mean(r1600), mean(m1600),
            mean(restCosts), mean(ceilingRecs)));
        out.append(String.format(Locale.ROOT,
            "%n[sequential marked restarts (path-marking, +100 dc budget each)]%n"
                + "  R=1 : recall=%.3f at mean cost %.0f dc%n"
                + "  R=16: recall=%.3f at mean cost %.0f dc%n",
            mean(seq1), mean(seqCost1s), mean(seq16), mean(seqCost16s)));
        out.append(String.format(Locale.ROOT,
            "%n[idea-3: free local multi-seed (descent 2nd/3rd best as beam seeds, same budget)]%n"));
        double ngBase = 0, ngLm = 0;
        int ngC = 0;
        for (int i = 0; i < classes.size(); i++) {
          if (!classes.get(i).equals("GOOD")) {
            ngBase += baseRecs.get(i);
            ngLm += lmRecs.get(i);
            ngC++;
          }
        }
        out.append(String.format(Locale.ROOT,
            "  non-good: single-seed=%.3f | 3-seed=%.3f | delta=%+.3f (D=%.0f vs %.0f)%n",
            ngC == 0 ? 0 : ngBase / ngC, ngC == 0 ? 0 : ngLm / ngC,
            ngC == 0 ? 0 : (ngLm - ngBase) / ngC, meanD(), mean(lmDs)));
      }

      // signal analysis: mean gain per bucket (non-good queries only)
      out.append(String.format("%n[signal -> ceiling gain correlation (non-good queries)]%n"));
      bucket(out, "lateFrac", 0, new double[]{0, 0.001, 0.2, 1.01},
          new String[]{"==0", "(0,0.2]", ">0.2"});
      bucket(out, "impCount", 1, new double[]{0, 5.1, 10.1, 20.1, 1e9},
          new String[]{"<=5", "6-10", "11-20", ">20"});
      bucket(out, "patienceRatio", 2, new double[]{0, 0.1, 0.3, 1.01},
          new String[]{"<=0.1", "(0.1,0.3]", ">0.3"});
    }

    void bucket(Appendable out, String signal, int idx, double[] edges, String[] labels)
        throws IOException {
      int[] cnt = new int[labels.length];
      double[] gsum = new double[labels.length];
      for (double[] row : signalRows) {
        for (int b = 0; b < labels.length; b++) {
          if (row[idx] >= edges[b] && row[idx] < edges[b + 1]) {
            cnt[b]++;
            gsum[b] += row[3];
            break;
          }
        }
      }
      out.append(String.format("  %-14s", signal));
      for (int b = 0; b < labels.length; b++) {
        out.append(String.format(Locale.ROOT, "  %s:n=%d gain=%.3f", labels[b], cnt[b],
            cnt[b] == 0 ? 0 : gsum[b] / cnt[b]));
      }
      out.append("\n");
    }

    static double mean(List<Double> xs) {
      double s = 0;
      for (double x : xs) s += x;
      return s / xs.size();
    }

    double meanD() {
      double s = 0;
      for (long x : ds) s += x;
      return s / ds.size();
    }
  }
}
