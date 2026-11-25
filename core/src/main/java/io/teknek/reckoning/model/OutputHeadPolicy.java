package io.teknek.reckoning.model;

import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.WeightLoader;

import java.util.Objects;

/** Selects the causal-LM output weight according to the model configuration. */
public final class OutputHeadPolicy {
    private OutputHeadPolicy() {
    }

    public static String weightName(Config config, WeightLoader weights,
            String inputEmbeddingName, String outputEmbeddingName) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(inputEmbeddingName, "inputEmbeddingName");
        Objects.requireNonNull(outputEmbeddingName, "outputEmbeddingName");
        if (config.tieWordEmbeddings()) {
            return inputEmbeddingName;
        }
        if (!weights.isWeightPresent(outputEmbeddingName)) {
            throw new IllegalArgumentException("Untied output head is missing: " + outputEmbeddingName);
        }
        return outputEmbeddingName;
    }
}
