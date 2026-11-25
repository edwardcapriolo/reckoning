package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.generator2.MLPBlock2;
import io.teknek.reckoning.generator2.RmsNorm2;
import io.teknek.reckoning.generator2.TransformerBlock2;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.tensor2.Lighter;

import java.util.Optional;

/** Spark transformer block composition using the shared residual block flow. */
final class Spark2_5TransformerBlock {
    private Spark2_5TransformerBlock() {
    }

    static TransformerBlock2 create(AbstractModel model, Spark2_5Config config, int layerIndex,
            Spark2_5LayerWeights weights, Lighter lighter) {
        String layerType = config.layerTypes.get(layerIndex);
        Spark2_5Attention attention = new Spark2_5Attention(model, config, layerIndex, layerType,
                weights.qkvProjection, weights.attentionGate, weights.outputProjection,
                new Spark2_5Rope(config, layerType, lighter), lighter);
        MLPBlock2 mlp = new MLPBlock2(model, weights.gateProjection, weights.upProjection,
                weights.downProjection, lighter);
        return new TransformerBlock2(model, layerIndex,
                Optional.of(new RmsNorm2(model, weights.inputLayerNorm, 0.0f)),
                attention,
                Optional.empty(),
                Optional.of(new RmsNorm2(model, weights.postAttentionLayerNorm, 0.0f)),
                mlp,
                Optional.empty(),
                Optional.empty());
    }
}
