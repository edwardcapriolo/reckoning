package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.generator2.ForwardContext;

/** Per-forward Spark state, matching the source model's position embeddings and cache position. */
public record Spark2_5ForwardContext(
        Spark2_5RopeCache ropeCache,
        int[] cachePosition,
        int[] attentionMask) implements ForwardContext {
}
