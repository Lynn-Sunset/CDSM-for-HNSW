package phase0;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.hnsw.HnswGraph;

import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Persistence for in-memory HNSW graphs (OnHeapHnswGraph): dumps the whole
 * neighbor structure via the public read API (seek/nextNeighbor per node per
 * level, getNodesOnLevel) and reloads it into a faithful array-backed
 * HnswGraph. Neighbor ORDER is preserved exactly (heap tie-breaking in the
 * instrumented searcher depends on insertion order), so a loaded graph
 * reproduces the original search results bit-for-bit.
 *
 * <p>Format: MAGIC, size, numLevels, entryNode; per level: nodeCount, node
 * ords, then per node: neighborCount, neighbor ords.
 *
 * <p>1M-node graph: ~150 MB on disk, ~2 s to load — replaces a ~20-min build.
 */
public final class PhaseHnswStore {

  static final int MAGIC = 0x48A45B;

  public static void dump(HnswGraph graph, Path file) throws IOException {
    int size = graph.size();
    int numLevels = graph.numLevels();
    int entry = graph.entryNode();
    long t0 = System.currentTimeMillis();
    try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(
        Files.newOutputStream(file), 1 << 20))) {
      out.writeInt(MAGIC);
      out.writeInt(size);
      out.writeInt(numLevels);
      out.writeInt(entry);
      for (int level = 0; level < numLevels; level++) {
        int[] nodes = HnswGraph.NodesIterator.getSortedNodes(graph.getNodesOnLevel(level));
        out.writeInt(nodes.length);
        for (int nd : nodes) {
          out.writeInt(nd);
        }
        List<Integer> nb = new ArrayList<>();
        for (int nd : nodes) {
          graph.seek(level, nd);
          nb.clear();
          int x;
          while ((x = graph.nextNeighbor()) != DocIdSetIterator.NO_MORE_DOCS) {
            nb.add(x);
          }
          out.writeInt(nb.size());
          for (int y : nb) {
            out.writeInt(y);
          }
        }
      }
    }
    System.out.printf("[hnsw-store] dumped size=%d levels=%d -> %s (%d ms)%n",
        size, numLevels, file, System.currentTimeMillis() - t0);
  }

  public static HnswGraph load(Path file) throws IOException {
    long t0 = System.currentTimeMillis();
    try (DataInputStream in = new DataInputStream(Files.newInputStream(file))) {
      if (in.readInt() != MAGIC) {
        throw new IOException("bad magic: " + file);
      }
      int size = in.readInt();
      int numLevels = in.readInt();
      int entry = in.readInt();
      int[][] levelNodes = new int[numLevels][];
      int[][][] neighbors = new int[numLevels][][];
      int[][] posOf = new int[numLevels][];
      for (int level = 0; level < numLevels; level++) {
        int cnt = in.readInt();
        int[] nodes = new int[cnt];
        for (int i = 0; i < cnt; i++) {
          nodes[i] = in.readInt();
        }
        levelNodes[level] = nodes;
        int[] pos = new int[size];
        java.util.Arrays.fill(pos, -1);
        for (int i = 0; i < cnt; i++) {
          pos[nodes[i]] = i;
        }
        posOf[level] = pos;
        int[][] nb = new int[cnt][];
        for (int i = 0; i < cnt; i++) {
          int nbc = in.readInt();
          int[] arr = new int[nbc];
          for (int j = 0; j < nbc; j++) {
            arr[j] = in.readInt();
          }
          nb[i] = arr;
        }
        neighbors[level] = nb;
      }
      HnswGraph g = new SimpleHnswGraph(size, numLevels, entry, levelNodes, neighbors, posOf);
      System.out.printf("[hnsw-store] loaded size=%d levels=%d from %s (%d ms)%n",
          size, numLevels, file, System.currentTimeMillis() - t0);
      return g;
    }
  }

  /** Array-backed HnswGraph; neighbor order preserved from the dump. */
  static final class SimpleHnswGraph extends HnswGraph {
    final int size;
    final int numLevels;
    final int entry;
    final int[][] levelNodes;
    final int[][][] neighbors;
    final int[][] posOf;
    int curLevel = -1;
    int curIdx = -1;
    int curPos = 0;

    SimpleHnswGraph(int size, int numLevels, int entry, int[][] levelNodes,
                    int[][][] neighbors, int[][] posOf) {
      this.size = size;
      this.numLevels = numLevels;
      this.entry = entry;
      this.levelNodes = levelNodes;
      this.neighbors = neighbors;
      this.posOf = posOf;
    }

    @Override
    public void seek(int level, int target) {
      curLevel = level;
      curPos = 0;
      curIdx = (level >= 0 && level < numLevels && target >= 0 && target < size)
          ? posOf[level][target] : -1;
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public int nextNeighbor() {
      if (curIdx < 0) {
        return DocIdSetIterator.NO_MORE_DOCS;
      }
      int[] nb = neighbors[curLevel][curIdx];
      if (curPos >= nb.length) {
        curIdx = -1;
        return DocIdSetIterator.NO_MORE_DOCS;
      }
      return nb[curPos++];
    }

    @Override
    public int numLevels() {
      return numLevels;
    }

    @Override
    public int entryNode() {
      return entry;
    }

    @Override
    public NodesIterator getNodesOnLevel(int level) {
      int[] nodes = levelNodes[level];
      return new NodesIterator(nodes.length) {
        int i = 0;

        @Override
        public boolean hasNext() {
          return i < size;
        }

        @Override
        public int nextInt() {
          return nodes[i++];
        }

        @Override
        public int consume(int[] dest) {
          int n = Math.min(dest.length, size - i);
          System.arraycopy(nodes, i, dest, 0, n);
          i += n;
          return n;
        }
      };
    }
  }
}
