package io.teknek.reckoning.model;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.model.spark.Spark2_5Config;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.tensor.TensorInfo;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class OutputHeadPolicyTest {
    @Test
    void tiedPolicyProjectsLogitsWithInputEmbeddingWeight() {
        Spark2_5Config config = config(true);
        WeightLoader weights = loader("model.embedding.weight", "lm_head.weight");
        assertProjection(config, weights, "model.embedding.weight", new float[][] {
                {1.0f, 0.0f}, {0.0f, 1.0f}, {2.0f, 3.0f}
        }, new float[] {4.0f, 5.0f}, new float[] {4.0f, 5.0f, 23.0f});
    }

    @Test
    void untiedPolicyRequiresAndUsesSeparateOutputWeight() {
        Spark2_5Config config = config(false);
        WeightLoader weights = loader("model.embedding.weight", "lm_head.weight");
        assertProjection(config, weights, "lm_head.weight", new float[][] {
                {10.0f, 0.0f}, {0.0f, 10.0f}, {1.0f, 1.0f}
        }, new float[] {4.0f, 5.0f}, new float[] {40.0f, 50.0f, 9.0f});

        WeightLoader missing = loader("model.embedding.weight");
        assertThrows(IllegalArgumentException.class,
                () -> OutputHeadPolicy.weightName(config, missing, "model.embedding.weight", "lm_head.weight"));
    }

    private static void assertProjection(Spark2_5Config config, WeightLoader weights, String expectedName,
            float[][] matrix, float[] hidden, float[] expected) {
        String selected = OutputHeadPolicy.weightName(config, weights, "model.embedding.weight", "lm_head.weight");
        assertEquals(expectedName, selected);
        Lighter lighter = new Lighter();
        try (TensorRef input = lighter.allocate(DType.F32, TensorShape.of(1, 2));
             TensorRef outputWeights = matrix(lighter, matrix);
             TensorRef logits = lighter.allocate(DType.F32, TensorShape.of(1, expected.length))) {
            input.set(hidden[0], 0, 0);
            input.set(hidden[1], 0, 1);
            lighter.dotProductRows(logits, input, outputWeights, 0, 0, 2, 0, expected.length, 0);
            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], logits.get(0, i), 1.0e-6f, "logit=" + i);
            }
        }
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

    private static WeightLoader loader(String... names) {
        Map<String, TensorInfo> tensorInfo = new java.util.HashMap<>();
        for (String name : names) tensorInfo.put(name, null);
        return new WeightLoader() {
            @Override public Map<String, String> metadata() { return Map.of(); }
            @Override public Map<String, TensorInfo> tensorInfoMap() { return tensorInfo; }
            @Override public DType getModelDType() { return DType.F32; }
            @Override public void close() { }
        };
    }

    private static Spark2_5Config config(boolean tied) {
        return new Spark2_5Config(
                List.of("Spark2_5ForCausalLM"), "spark2_5", Map.of(), "float32",
                false, 0.0f, 0, 1, "sigmoid", 2, false, "gelu", 2, 0.0221f, 1,
                List.of("full_attention"), 8, false, 1, 1, 1, 2, 1.0e-6f,
                Map.of("sliding_attention", Map.of("rope_theta", 10_000.0, "partial_rotary_factor", 1.0),
                        "full_attention", Map.of("rope_theta", 5_000_000.0, "partial_rotary_factor", 0.25)),
                512, tied, true, 16);
    }
}
