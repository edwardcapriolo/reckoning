package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.ForwardPhase;
import io.teknek.reckoning.generator2.MLPBlock2;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Characterization test for the first Spark forward component.
 *
 * <p>The fixture uses the existing tensor-native {@link MLPBlock2} with a
 * mocked model shell. This keeps the characterization independent of Spark
 * attention and cache work while proving that a GELU-configured model executes
 * the Spark MLP ordering.</p>
 */
class Spark2_5MlpCharacterizationTest {
    @Test
    void sparkUsesGeluGatedMlpOrdering() {
        float input0 = 1.0f;
        float input1 = 2.0f;

        Spark2_5TestHarness.Fixture fixture = Spark2_5TestHarness.model(Spark2_5TestHarness.config(true));
        AbstractModel model = fixture.model();
        Lighter lighter = fixture.lighter();

        try (TensorRef input = matrix(lighter, new float[][] {{input0, input1}});
             TensorRef gate = matrix(lighter, new float[][] {{1.0f, 0.0f}});
             TensorRef up = matrix(lighter, new float[][] {{2.0f, 0.0f}});
             TensorRef down = matrix(lighter, new float[][] {{3.0f}, {0.0f}})) {
            MLPBlock2 block = new MLPBlock2(model, gate, up, down, lighter);
            try (TensorRef output = block.forward(input, Optional.empty(), ForwardPhase.PREFILL)) {
                float expected = 3.0f * (gelu(input0) * 2.0f);
                assertEquals(expected, output.get(0, 0), 1.0e-5f);
                assertEquals(0.0f, output.get(0, 1), 1.0e-6f);
            }
        }
    }

    /** Independent GELU reference used only to state the expected fixture value. */
    private static float gelu(float value) {
        return (float) (0.5 * value * (1.0 + erf(value / Math.sqrt(2.0))));
    }

    private static double erf(double value) {
        double sign = Math.copySign(1.0, value);
        double absolute = Math.abs(value);
        double t = 1.0 / (1.0 + 0.3275911 * absolute);
        double result = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * Math.exp(-absolute * absolute);
        return sign * result;
    }

    private static TensorRef matrix(Lighter lighter, float[][] values) {
        TensorRef result = lighter.allocate(DType.F32, TensorShape.of(values.length, values[0].length));
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                result.set(values[row][column], row, column);
            }
        }
        return result;
    }
}
