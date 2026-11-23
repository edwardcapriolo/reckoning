package io.teknek.reckoning.tensor2;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HeadwiseSigmoidGateFuzzTest {
    @Test
    void naiveAndPanamaAgreeAcrossShapesTailsAndGateRanges() {
        Random random = new Random(0x535041524B474154L);
        for (int iteration = 0; iteration < 200; iteration++) {
            int rows = 1 + random.nextInt(12);
            int heads = 1 + random.nextInt(8);
            int headSize = 1 + random.nextInt(37);
            assertCase(rows, heads, headSize, random);
        }
    }

    private static void assertCase(int rows, int heads, int headSize, Random random) {
        Lighter lighter = new Lighter();
        int columns = heads * headSize;
        float[][] originalScores = new float[rows][heads];
        try (TensorRef naiveTarget = lighter.allocate(DType.F32, TensorShape.of(rows, columns));
             TensorRef panamaTarget = lighter.allocate(DType.F32, TensorShape.of(rows, columns));
             TensorRef naiveGates = lighter.allocate(DType.F32, TensorShape.of(rows, heads));
             TensorRef panamaGates = lighter.allocate(DType.F32, TensorShape.of(rows, heads))) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    float value = (random.nextFloat() - 0.5f) * 20.0f;
                    naiveTarget.set(value, row, column);
                    panamaTarget.set(value, row, column);
                }
                for (int head = 0; head < heads; head++) {
                    float score = (random.nextFloat() - 0.5f) * 40.0f;
                    originalScores[row][head] = score;
                    naiveGates.set(score, row, head);
                    panamaGates.set(score, row, head);
                }
            }

            NaiveOps naive = new NaiveOps();
            PanamaOps panama = new PanamaOps();
            assertTrue(naive.sigmoid(naiveGates).isRight());
            assertTrue(panama.sigmoid(panamaGates).isRight());
            assertTrue(naive.headwiseMultiply(new HeadwiseMultiply(naiveTarget, naiveGates, heads, headSize)).isRight());
            assertTrue(panama.headwiseMultiply(new HeadwiseMultiply(panamaTarget, panamaGates, heads, headSize)).isRight());

            for (int row = 0; row < rows; row++) {
                for (int head = 0; head < heads; head++) {
                    float gate = naiveGates.get(row, head);
                    assertEquals(gate, panamaGates.get(row, head), 2.0e-6f,
                            "gate row=" + row + " head=" + head);
                    assertEquals(1.0f / (1.0f + (float) Math.exp(-originalScores[row][head])), gate, 2.0e-5f,
                            "reference gate row=" + row + " head=" + head);
                    for (int dimension = 0; dimension < headSize; dimension++) {
                        int column = head * headSize + dimension;
                        assertEquals(naiveTarget.get(row, column), panamaTarget.get(row, column), 2.0e-5f,
                                "row=" + row + " head=" + head + " dimension=" + dimension);
                    }
                }
            }
        }
    }
}
