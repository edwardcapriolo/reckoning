package io.teknek.reckoning.tensor2;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PanamaSaxpyDTypeTest {
    @Test
    void supportsScalarF32AlphaBf16ValuesAndBf16Output() {
        Lighter lighter = new Lighter();
        PanamaOps ops = new PanamaOps();
        try (TensorRef values = lighter.allocate(DType.BF16, TensorShape.of(1, 4));
             TensorRef output = lighter.allocate(DType.BF16, TensorShape.of(1, 4))) {
            for (int column = 0; column < 4; column++) {
                values.set(column + 1.0f, 0, column);
                output.set(1.0f, 0, column);
            }
            assertTrue(ops.saxpy(2.0f, values, output, 0, 0, 4).isRight());
            for (int column = 0; column < 4; column++) {
                assertEquals(1.0f + 2.0f * (column + 1.0f), output.get(0, column), 0.05f);
            }
        }
    }

    @Test
    void supportsF32AlphaBf16ValuesAndBf16Output() {
        Lighter lighter = new Lighter();
        PanamaOps ops = new PanamaOps();
        try (TensorRef alpha = lighter.allocate(DType.F32, TensorShape.of(1, 2));
             TensorRef values = lighter.allocate(DType.BF16, TensorShape.of(2, 4));
             TensorRef output = lighter.allocate(DType.BF16, TensorShape.of(1, 4))) {
            alpha.set(0.5f, 0, 0);
            alpha.set(2.0f, 0, 1);
            for (int column = 0; column < 4; column++) {
                values.set(column + 1.0f, 0, column);
                values.set(column + 5.0f, 1, column);
                output.set(1.0f, 0, column);
            }

            assertTrue(ops.saxpy(alpha, values, output, 0, 0, 4, 0, 0, 2).isRight());
            for (int column = 0; column < 4; column++) {
                assertEquals(1.0f + 0.5f * (column + 1.0f) + 2.0f * (column + 5.0f),
                        output.get(0, column), 0.05f);
            }
        }
    }

    @Test
    void supportsF32ValuesAndBf16Output() {
        Lighter lighter = new Lighter();
        PanamaOps ops = new PanamaOps();
        try (TensorRef values = lighter.allocate(DType.F32, TensorShape.of(1, 4));
             TensorRef output = lighter.allocate(DType.BF16, TensorShape.of(1, 4))) {
            for (int column = 0; column < 4; column++) {
                values.set(column + 1.0f, 0, column);
                output.set(1.0f, 0, column);
            }

            assertTrue(ops.saxpy(2.0f, values, output, 0, 0, 4).isRight());
            for (int column = 0; column < 4; column++) {
                assertEquals(1.0f + 2.0f * (column + 1.0f), output.get(0, column), 0.05f);
            }
        }
    }

    @Test
    void supportsVectorizedBf16ValuesAndBf16Output() {
        Lighter lighter = new Lighter();
        PanamaOps ops = new PanamaOps();
        try (TensorRef values = lighter.allocate(DType.BF16, TensorShape.of(1, 8));
             TensorRef output = lighter.allocate(DType.BF16, TensorShape.of(1, 8))) {
            for (int column = 0; column < 8; column++) {
                values.set(column + 1.0f, 0, column);
                output.set(1.0f, 0, column);
            }

            assertTrue(ops.saxpy(2.0f, values, output, 0, 0, 8).isRight());
            for (int column = 0; column < 8; column++) {
                assertEquals(1.0f + 2.0f * (column + 1.0f), output.get(0, column), 0.05f);
            }
        }
    }

    @Test
    void supportsVectorizedBf16ValuesAndF32Output() {
        Lighter lighter = new Lighter();
        PanamaOps ops = new PanamaOps();
        try (TensorRef values = lighter.allocate(DType.BF16, TensorShape.of(1, 8));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(1, 8))) {
            for (int column = 0; column < 8; column++) {
                values.set(column + 1.0f, 0, column);
                output.set(1.0f, 0, column);
            }

            assertTrue(ops.saxpy(2.0f, values, output, 0, 0, 8).isRight());
            for (int column = 0; column < 8; column++) {
                assertEquals(1.0f + 2.0f * (column + 1.0f), output.get(0, column), 0.05f);
            }
        }
    }

    @Test
    void supportsVectorizedBatchBf16ValuesAndF32Output() {
        Lighter lighter = new Lighter();
        PanamaOps ops = new PanamaOps();
        try (TensorRef alpha = lighter.allocate(DType.F32, TensorShape.of(1, 1));
             TensorRef values = lighter.allocate(DType.BF16, TensorShape.of(1, 8));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(1, 8))) {
            alpha.set(2.0f, 0, 0);
            for (int column = 0; column < 8; column++) {
                values.set(column + 1.0f, 0, column);
                output.set(1.0f, 0, column);
            }

            assertTrue(ops.saxpy(alpha, values, output, 0, 0, 8, 0, 0, 1).isRight());
            for (int column = 0; column < 8; column++) {
                assertEquals(1.0f + 2.0f * (column + 1.0f), output.get(0, column), 0.05f);
            }
        }
    }

    @Test
    void supportsVectorizedBatchBf16AlphaAndValuesAndF32Output() {
        Lighter lighter = new Lighter();
        PanamaOps ops = new PanamaOps();
        try (TensorRef alpha = lighter.allocate(DType.BF16, TensorShape.of(1, 1));
             TensorRef values = lighter.allocate(DType.BF16, TensorShape.of(1, 8));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(1, 8))) {
            alpha.set(2.0f, 0, 0);
            for (int column = 0; column < 8; column++) {
                values.set(column + 1.0f, 0, column);
                output.set(1.0f, 0, column);
            }

            assertTrue(ops.saxpy(alpha, values, output, 0, 0, 8, 0, 0, 1).isRight());
            for (int column = 0; column < 8; column++) {
                assertEquals(1.0f + 2.0f * (column + 1.0f), output.get(0, column), 0.05f);
            }
        }
    }
}
