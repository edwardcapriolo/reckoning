package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.tensor.kv.KvCacheSession;
import io.teknek.reckoning.tensor2.TensorRef;

/** Spark counterpart of the source model's BaseModelOutputWithPast. */
public record Spark2_5ModelOutput(
        TensorRef lastHiddenState,
        KvCacheSession pastKeyValues) {
}
