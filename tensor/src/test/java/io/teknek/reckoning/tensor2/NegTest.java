package io.teknek.reckoning.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorDisplayUtil;
import io.teknek.reckoning.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class NegTest {
    @Test
    void panamaMatchesNaiveAcrossF32AndBf16Shapes() {
        Random random = new Random(0x4E4547L);
        int mismatches = 0;
        for (DType dType : new DType[]{DType.F32, DType.BF16}) {
            for (int iteration = 0; iteration < 200; iteration++) {
                int rows = 1 + random.nextInt(8);
                int columns = 1 + random.nextInt(40);
                Lighter lighter = new Lighter(new MetricRegistry());
                try (TensorRef source = lighter.allocate(dType, TensorShape.of(rows, columns));
                     TensorRef naiveOutput = lighter.allocate(dType, TensorShape.of(rows, columns));
                     TensorRef panamaOutput = lighter.allocate(dType, TensorShape.of(rows, columns))) {
                    for (int row = 0; row < rows; row++) {
                        for (int column = 0; column < columns; column++) {
                            source.set((random.nextFloat() - 0.5f) * 200.0f, row, column);
                        }
                    }
                    new NaiveOps().neg(source, naiveOutput);
                    new PanamaOps().neg(source, panamaOutput);
                    for (int row = 0; row < rows; row++) {
                        for (int column = 0; column < columns; column++) {
                            if (Float.floatToIntBits(naiveOutput.get(row, column))
                                    != Float.floatToIntBits(panamaOutput.get(row, column))) {
                                mismatches++;
                            }
                        }
                    }
                }
            }
        }
        assertEquals(0, mismatches, "Panama neg must match the naive provider");
    }

    @Test
    void displaysNegatedTensor2dResult() {
        Lighter lighter = new Lighter();
        try (TensorRef source = lighter.allocate(DType.F32, TensorShape.of(2, 4));
             TensorRef result = lighter.allocate(DType.F32, TensorShape.of(2, 4))) {
            float[] values = {1.5f, -2.0f, 0.0f, 4.25f, -5.5f, 6.0f, -7.75f, 8.0f};
            for (int index = 0; index < values.length; index++) {
                source.set(values[index], index / 4, index % 4);
            }
            lighter.neg(new Neg(source).into(result));

            assertEquals("[0][0]= -1.5000 [0][1]=  2.0000 [0][2]= -0.0000 [0][3]= -4.2500 \n"
                            + "[1][0]=  5.5000 [1][1]= -6.0000 [1][2]=  7.7500 [1][3]= -8.0000 \n",
                    TensorDisplayUtil.pretty2dDisplayAll(new TensorRefBackedTensor(result)));
        }
    }
}
