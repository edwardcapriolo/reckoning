package io.teknek.reckoning.tensor2;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PanamaHeadwiseMultiplyBf16FuzzTest {
    @Test
    void panamaMatchesNaiveReferenceAcrossRandomBf16Cases() {
        Random random = new Random(0x535041524B424631L);
        int mismatches = 0;
        for (int iteration = 0; iteration < 200; iteration++) {
            int rows = 1 + random.nextInt(8);
            int heads = 1 + random.nextInt(8);
            int headSize = 8 + random.nextInt(30);
            int width = heads * headSize;
            Lighter lighter = new Lighter();
            try (TensorRef naiveTarget = lighter.allocate(DType.BF16, TensorShape.of(rows, width));
                 TensorRef panamaTarget = lighter.allocate(DType.BF16, TensorShape.of(rows, width));
                 TensorRef naiveGates = lighter.allocate(DType.BF16, TensorShape.of(rows, heads));
                 TensorRef panamaGates = lighter.allocate(DType.BF16, TensorShape.of(rows, heads))) {
                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < width; column++) {
                        float value = (random.nextFloat() - 0.5f) * 20.0f;
                        naiveTarget.set(value, row, column);
                        panamaTarget.set(value, row, column);
                    }
                    for (int head = 0; head < heads; head++) {
                        float gate = (random.nextFloat() - 0.5f) * 4.0f;
                        naiveGates.set(gate, row, head);
                        panamaGates.set(gate, row, head);
                    }
                }
                new NaiveOps().headwiseMultiply(new HeadwiseMultiply(naiveTarget, naiveGates, heads, headSize));
                new PanamaOps().headwiseMultiply(new HeadwiseMultiply(panamaTarget, panamaGates, heads, headSize));
                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < width; column++) {
                        if (Float.floatToIntBits(naiveTarget.get(row, column))
                                != Float.floatToIntBits(panamaTarget.get(row, column))) {
                            mismatches++;
                        }
                    }
                }
            }
        }
        assertEquals(0, mismatches, "Panama BF16 vector store must match the naive provider");
    }
}
