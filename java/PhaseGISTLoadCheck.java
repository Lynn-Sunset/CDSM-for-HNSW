package phase0;

import java.nio.file.Paths;

/** Quick check of PhaseGISTIndex.loadFvecs row 0 vs expected values. */
public final class PhaseGISTLoadCheck {
  public static void main(String[] args) throws Exception {
    float[][] base = PhaseGISTIndex.loadFvecs(Paths.get(args[0]));
    System.out.printf("rows=%d dim=%d%n", base.length, base[0].length);
    System.out.printf("row0[0..4] = %f %f %f %f %f%n",
        base[0][0], base[0][1], base[0][2], base[0][3], base[0][4]);
    // row 786559 (true neighbor of query 0, expected sim 0.638)
    System.out.printf("row786559[0..4] = %f %f %f %f %f%n",
        base[786559][0], base[786559][1], base[786559][2], base[786559][3], base[786559][4]);
  }
}
