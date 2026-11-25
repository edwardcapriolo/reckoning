package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.tensor2.TensorRef;

/** Loaded tensors belonging to one Spark decoder layer. */
final class Spark2_5LayerWeights {
    final TensorRef inputLayerNorm;
    final TensorRef postAttentionLayerNorm;
    final TensorRef qkvProjection;
    final TensorRef attentionGate;
    final TensorRef outputProjection;
    final TensorRef gateProjection;
    final TensorRef upProjection;
    final TensorRef downProjection;

    Spark2_5LayerWeights(TensorRef inputLayerNorm, TensorRef postAttentionLayerNorm,
            TensorRef qkvProjection, TensorRef attentionGate, TensorRef outputProjection,
            TensorRef gateProjection, TensorRef upProjection, TensorRef downProjection) {
        this.inputLayerNorm = inputLayerNorm;
        this.postAttentionLayerNorm = postAttentionLayerNorm;
        this.qkvProjection = qkvProjection;
        this.attentionGate = attentionGate;
        this.outputProjection = outputProjection;
        this.gateProjection = gateProjection;
        this.upProjection = upProjection;
        this.downProjection = downProjection;
    }
}
