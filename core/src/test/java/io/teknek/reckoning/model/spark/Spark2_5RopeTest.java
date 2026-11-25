package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Spark2_5RopeTest {
    @Test
    void slidingRotatesAllDimensionsAndFullRotatesOnlyFirst64() {
        Spark2_5Config config = Spark2_5TestHarness.realShapeConfig();
        Lighter lighter = new Lighter();
        try (Spark2_5Rope sliding = new Spark2_5Rope(config, "sliding_attention", lighter);
             Spark2_5Rope full = new Spark2_5Rope(config, "full_attention", lighter);
             TensorRef slidingTarget = target(lighter);
             TensorRef fullTarget = target(lighter)) {
            sliding.apply(slidingTarget, 8, 7);
            full.apply(fullTarget, 8, 7);

            for (int head = 0; head < 8; head++) {
                int offset = head * 256;
                assertNotEqual(offset + 65.0f, slidingTarget.get(0, offset + 64));
                assertEquals(offset + 65.0f, fullTarget.get(0, offset + 64), 1.0e-6f,
                        "full RoPE tail must pass through unchanged head=" + head);
                assertEquals(offset + 256.0f, fullTarget.get(0, offset + 255), 1.0e-6f,
                        "full RoPE tail must pass through unchanged head=" + head);
            }

            double angle = 7.0;
            assertEquals(Math.cos(angle) - 33.0 * Math.sin(angle),
                    fullTarget.get(0, 0), 1.0e-4f);
            assertEquals(Math.sin(angle) + 33.0 * Math.cos(angle),
                    fullTarget.get(0, 32), 1.0e-4f);
            assertEquals(Math.cos(angle) - 129.0 * Math.sin(angle),
                    slidingTarget.get(0, 0), 1.0e-4f);
            assertEquals(Math.sin(angle) + 129.0 * Math.cos(angle),
                    slidingTarget.get(0, 128), 1.0e-4f);
        }
    }

    private static TensorRef target(Lighter lighter) {
        TensorRef result = lighter.allocate(DType.F32, TensorShape.of(1, 8 * 256));
        for (int dimension = 0; dimension < 8 * 256; dimension++) {
            result.set(dimension + 1.0f, 0, dimension);
        }
        return result;
    }

    private static void assertNotEqual(float expected, float actual) {
        if (Math.abs(expected - actual) < 1.0e-4f) {
            throw new AssertionError("expected rotation to change value " + expected);
        }
    }
}
