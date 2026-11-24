package io.teknek.reckoning.tensor2;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeBf16F32DotProductRowsFuzzTest {
    @Test
    void matchesPanamaAcrossRowsOffsetsAndTails() {
        Assumptions.assumeTrue(NativeOps.isAvailable());
        Random random = new Random(0x42463136444F54L);
        NativeOps nativeOps = new NativeOps();
        PanamaOps panamaOps = new PanamaOps();
        Lighter lighter = new Lighter();
        try {
            for (int iteration = 0; iteration < 200; iteration++) {
                int resultRows = 1 + random.nextInt(4);
                int inputColumns = 64 + random.nextInt(193);
                int weightColumns = inputColumns + random.nextInt(17);
                int weightRows = 1 + random.nextInt(8);
                int inputStart = random.nextInt(Math.min(32, inputColumns - 1));
                int weightStart = random.nextInt(Math.min(32, weightColumns - 1));
                int columnLength = 1 + random.nextInt(Math.min(inputColumns - inputStart,
                        weightColumns - weightStart));
                int weightRowStart = random.nextInt(weightRows);
                int weightRowCount = 1 + random.nextInt(weightRows - weightRowStart);
                int outputColumnStart = random.nextInt(5);
                try (TensorRef input = lighter.allocate(DType.BF16, TensorShape.of(resultRows, inputColumns));
                     TensorRef weights = lighter.allocate(DType.F32, TensorShape.of(weightRows, weightColumns));
                     TensorRef nativeOutput = lighter.allocate(DType.F32,
                             TensorShape.of(resultRows, outputColumnStart + weightRowCount + 2));
                     TensorRef panamaOutput = lighter.allocate(DType.F32,
                             TensorShape.of(resultRows, outputColumnStart + weightRowCount + 2))) {
                    fill(input, random);
                    fill(weights, random);
                    assertTrue(nativeOps.dotProductRows(nativeOutput, input, weights, inputStart, weightStart,
                            columnLength, weightRowStart, weightRowCount, outputColumnStart).isRight());
                    assertTrue(panamaOps.dotProductRows(panamaOutput, input, weights, inputStart, weightStart,
                            columnLength, weightRowStart, weightRowCount, outputColumnStart).isRight());
                    for (int row = 0; row < resultRows; row++) {
                        for (int column = 0; column < weightRowCount; column++) {
                            assertEquals(panamaOutput.get(row, outputColumnStart + column),
                                    nativeOutput.get(row, outputColumnStart + column), 0.02f,
                                    "iteration=" + iteration + " row=" + row + " column=" + column);
                        }
                    }
                }
            }
        } finally {
            nativeOps.close();
        }
    }

    private static void fill(TensorRef tensor, Random random) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set((random.nextFloat() - 0.5f) * 4.0f, row, column);
            }
        }
    }
}
