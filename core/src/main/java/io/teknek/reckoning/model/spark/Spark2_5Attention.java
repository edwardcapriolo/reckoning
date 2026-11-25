package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.CompositeOps;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensor2.HeadwiseSigmoidGate;
import io.teknek.reckoning.generator2.ForwardContext;
import io.teknek.reckoning.generator2.SelfAttention2;
import io.teknek.reckoning.generator2.PackedBlockAttention2;
import io.teknek.reckoning.generator.ForwardPhase;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.tensor.kv.CacheExecutionMode;
import io.teknek.reckoning.tensor.kv.KvCacheSession;
import io.teknek.reckoning.tensor.kv.KvReadView;
import io.teknek.reckoning.tensor.kv.KvWriteCursor;
import io.teknek.reckoning.tensor.kv.AttentionPattern;
import io.teknek.reckoning.tensor.TensorShape;

import java.util.Objects;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** Spark attention shell; forward behavior is implemented incrementally. */
public final class Spark2_5Attention implements SelfAttention2 {
    private final AbstractModel model;
    private final Spark2_5Config config;
    private final int layerIndex;
    private final String layerType;
    private final TensorRef qkvProjection;
    private final TensorRef attentionGate;
    private final TensorRef outputProjection;
    private final Spark2_5Rope rope;
    private final Lighter lighter;
    private final PackedBlockAttention2 attentionKernel;
    private final CompositeOps compositeOps;

    public Spark2_5Attention(AbstractModel model, Spark2_5Config config, int layerIndex, String layerType,
            TensorRef qkvProjection, TensorRef attentionGate, TensorRef outputProjection,
            Spark2_5Rope rope, Lighter lighter) {
        Objects.requireNonNull(model, "model");
        this.model = model;
        this.config = Objects.requireNonNull(config, "config");
        this.layerIndex = layerIndex;
        this.layerType = Objects.requireNonNull(layerType, "layerType");
        this.qkvProjection = Objects.requireNonNull(qkvProjection, "qkvProjection");
        this.attentionGate = Objects.requireNonNull(attentionGate, "attentionGate");
        this.outputProjection = Objects.requireNonNull(outputProjection, "outputProjection");
        this.rope = Objects.requireNonNull(rope, "rope");
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.attentionKernel = new PackedBlockAttention2(model, model.getMetricRegistry(), layerIndex);
        this.compositeOps = new CompositeOps(lighter);
    }

    public TensorRef forward(TensorRef hiddenStates, int startPosition) {
        return forwardUncached(hiddenStates, startPosition, null);
    }

    private TensorRef forwardUncached(TensorRef hiddenStates,
            int startPosition, Spark2_5RopeCache.PositionEmbeddings embeddings) {
        try (TensorRef fusedQkv = projectFusedQkv(hiddenStates)) {
            try (Spark2_5AttentionLayout.Split split = Spark2_5AttentionLayout.splitQkv(
                    fusedQkv,
                    config.numberOfHeads,
                    config.numberOfKeyValueHeads,
                    config.headSize,
                    lighter)) {
                applyRope(split.query(), config.numberOfHeads, startPosition, embeddings);
                applyRope(split.key(), config.numberOfKeyValueHeads, startPosition, embeddings);
                TensorRef attentionOutput = lighter.allocate(DType.F32, TensorShape.of(
                        hiddenStates.shape().first(), config.numberOfHeads * config.headSize));
                try {
                    int slidingWindow = "sliding_attention".equals(layerType)
                            ? Objects.requireNonNull(config.slidingWindow, "slidingWindow") : 0;
                    attentionKernel.forward(attentionOutput, split.query(), split.key(), split.value(),
                            0, hiddenStates.shape().first(), config.numberOfHeads,
                            config.numberOfKeyValueHeads, config.headSize,
                            (float) (1.0 / Math.sqrt(config.headSize)), null, true, slidingWindow);
                        try (TensorRef gateScores = lighter.allocate(DType.F32, TensorShape.of(
                            hiddenStates.shape().first(), config.numberOfHeads))) {
                        TensorRef gateInput = linearInputForWeights(hiddenStates, attentionGate);
                        try {
                            lighter.dotProductRows(gateScores, gateInput, attentionGate,
                                    0, 0, hiddenStates.shape().last(), 0, config.numberOfHeads, 0);
                        } finally {
                            if (gateInput != hiddenStates) gateInput.close();
                        }
                        model.emitLayerDebug(layerIndex, "g_proj", gateScores);
                        compositeOps.headwiseSigmoidGate(new HeadwiseSigmoidGate(
                                attentionOutput, gateScores, config.numberOfHeads, config.headSize));
                    }
                     TensorRef output = lighter.allocate(DType.F32, TensorShape.of(
                             hiddenStates.shape().first(), config.embeddingLength));
                     TensorRef outputInput = linearInputForWeights(attentionOutput, outputProjection);
                     try {
                         lighter.dotProductRows(output, outputInput, outputProjection,
                                 0, 0, attentionOutput.shape().last(), 0, config.embeddingLength, 0);
                         return output;
                     } catch (RuntimeException | Error error) {
                         output.close();
                         throw error;
                     } finally {
                         if (outputInput != attentionOutput) outputInput.close();
                     }
                } finally {
                    attentionOutput.close();
                }
            }
        }
    }

    @Override
    public TensorRef forward(TensorRef hiddenStates, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase,
            int batchSize, int sequenceLength, int[] attentionMask) {
        if (attentionMask != null) {
            throw new UnsupportedOperationException("Spark external attention masks are not implemented yet");
        }
        if (kvSession == null) return forward(hiddenStates, startPosition);
        CacheExecutionMode mode = startPosition == 0
                ? CacheExecutionMode.PREFILL_UPDATE_CACHE
                : batchSize == 1 ? CacheExecutionMode.DECODE_UPDATE_CACHE : CacheExecutionMode.PREFILL_UPDATE_CACHE;
        return forwardWithCache(hiddenStates, startPosition, kvSession, mode, null);
    }

    @Override
    public TensorRef forward(TensorRef hiddenStates, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase, int batchSize,
            int sequenceLength, int[] attentionMask, ForwardContext context) {
        Spark2_5ForwardContext sparkContext = (Spark2_5ForwardContext) context;
        if (attentionMask != null) {
            throw new UnsupportedOperationException("Spark external attention masks are not implemented yet");
        }
        Spark2_5RopeCache.PositionEmbeddings embeddings = sparkContext.ropeCache()
                .forLayerType(layerType);
        if (kvSession == null) return forwardUncached(hiddenStates, startPosition, embeddings);
        CacheExecutionMode mode = startPosition == 0
                ? CacheExecutionMode.PREFILL_UPDATE_CACHE
                : batchSize == 1 ? CacheExecutionMode.DECODE_UPDATE_CACHE : CacheExecutionMode.PREFILL_UPDATE_CACHE;
        return forwardWithCache(hiddenStates, startPosition, kvSession, mode, embeddings);
    }

    @Override
    public TensorRef forward(TensorRef hiddenStates, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase,
            CacheExecutionMode mode) {
        if (kvSession == null || mode == null) return forward(hiddenStates, startPosition);
        return forwardWithCache(hiddenStates, startPosition, kvSession, mode, null);
    }

    private TensorRef forwardWithCache(TensorRef hiddenStates, int startPosition,
            KvCacheSession kvSession, CacheExecutionMode mode,
            Spark2_5RopeCache.PositionEmbeddings embeddings) {
        try (TensorRef fusedQkv = projectFusedQkv(hiddenStates);
             Spark2_5AttentionLayout.Split split = Spark2_5AttentionLayout.splitQkv(
                     fusedQkv, config.numberOfHeads, config.numberOfKeyValueHeads, config.headSize, lighter)) {
            applyRope(split.query(), config.numberOfHeads, startPosition, embeddings);
            applyRope(split.key(), config.numberOfKeyValueHeads, startPosition, embeddings);
            if (writesCache(mode)) writeCurrentRows(kvSession, mode, split.key(), split.value(), startPosition);

            TensorRef attentionOutput = lighter.allocate(DType.F32, TensorShape.of(
                    hiddenStates.shape().first(), config.numberOfHeads * config.headSize));
            try {
                int slidingWindow = "sliding_attention".equals(layerType)
                        ? Objects.requireNonNull(config.slidingWindow, "slidingWindow") : 0;
                if (startPosition == 0) {
                    attentionKernel.forward(attentionOutput, split.query(), split.key(), split.value(), 0,
                            hiddenStates.shape().first(), config.numberOfHeads, config.numberOfKeyValueHeads,
                            config.headSize, (float) (1.0 / Math.sqrt(config.headSize)), null, true, slidingWindow);
                } else {
                    try (KvReadView prefix = kvSession.readView(layerIndex, startPosition, AttentionPattern.CAUSAL)) {
                        attentionKernel.forward(attentionOutput, split.query(), prefix, split.key(), split.value(),
                                startPosition, hiddenStates.shape().first(), config.numberOfHeads,
                                config.numberOfKeyValueHeads, config.headSize,
                                (float) (1.0 / Math.sqrt(config.headSize)), null, true, slidingWindow);
                    }
                }
                return projectAttentionOutput(attentionOutput, hiddenStates);
            } finally {
                attentionOutput.close();
            }
        }
    }

    private void applyRope(TensorRef target, int heads, int startPosition,
            Spark2_5RopeCache.PositionEmbeddings embeddings) {
        if (embeddings == null) {
            rope.apply(target, heads, startPosition);
        } else {
            rope.apply(target, heads, startPosition, embeddings);
        }
    }

    private void writeCurrentRows(KvCacheSession session, CacheExecutionMode mode,
            TensorRef keys, TensorRef values, int startPosition) {
        try (KvWriteCursor writer = session.writer(mode)) {
            for (int row = 0; row < keys.shape().first(); row++) {
                try (TensorRef key = keys.slice(row); TensorRef value = values.slice(row)) {
                    writer.write(layerIndex, startPosition + row, key, value);
                }
            }
        }
    }

    private boolean writesCache(CacheExecutionMode mode) {
        return mode != CacheExecutionMode.READ_PREFIX_NO_UPDATE
                && mode != CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE;
    }

    private TensorRef projectAttentionOutput(TensorRef attentionOutput, TensorRef hiddenStates) {
        try (TensorRef gateScores = lighter.allocate(DType.F32, TensorShape.of(
                hiddenStates.shape().first(), config.numberOfHeads))) {
            TensorRef gateInput = linearInputForWeights(hiddenStates, attentionGate);
            try {
                lighter.dotProductRows(gateScores, gateInput, attentionGate,
                        0, 0, hiddenStates.shape().last(), 0, config.numberOfHeads, 0);
            } finally {
                if (gateInput != hiddenStates) gateInput.close();
            }
            model.emitLayerDebug(layerIndex, "g_proj", gateScores);
            compositeOps.headwiseSigmoidGate(new HeadwiseSigmoidGate(
                    attentionOutput, gateScores, config.numberOfHeads, config.headSize));
        }
        TensorRef output = lighter.allocate(DType.F32, TensorShape.of(
                hiddenStates.shape().first(), config.embeddingLength));
        TensorRef outputInput = linearInputForWeights(attentionOutput, outputProjection);
        try {
            lighter.dotProductRows(output, outputInput, outputProjection,
                    0, 0, attentionOutput.shape().last(), 0, config.embeddingLength, 0);
            model.emitLayerDebug(layerIndex, "attention_output", output);
            return output;
        } catch (RuntimeException | Error error) {
            output.close();
            throw error;
        } finally {
            if (outputInput != attentionOutput) outputInput.close();
        }
    }

    /** Computes the single fused QKV projection; splitting is the next attention step. */
    TensorRef projectFusedQkv(TensorRef hiddenStates) {
        Objects.requireNonNull(hiddenStates, "hiddenStates");
        if (hiddenStates.dims() != 2
                || hiddenStates.shape().last() != config.embeddingLength
                || qkvProjection.dims() != 2
                || qkvProjection.shape().last() != config.embeddingLength
                || qkvProjection.shape().first() != config.numberOfHeads * config.headSize
                        + 2 * config.numberOfKeyValueHeads * config.headSize) {
            throw new IllegalArgumentException("Invalid Spark fused QKV projection shapes");
        }
        TensorRef fused = lighter.allocate(qkvProjection.dType(), TensorShape.of(
                hiddenStates.shape().first(), qkvProjection.shape().first()));
        try {
            lighter.dotProductRows(fused, hiddenStates, qkvProjection,
                    0, 0, hiddenStates.shape().last(), 0, qkvProjection.shape().first(), 0);
            model.emitLayerDebug(layerIndex, "q_k_v_proj", fused);
            return fused;
        } catch (RuntimeException | Error error) {
            if (fused != null) fused.close();
            throw error;
        }
    }

    private TensorRef linearInputForWeights(TensorRef input, TensorRef weights) {
        if (input.dType() == weights.dType()) return input;
        return lighter.reshape(input, weights.dType());
    }

}
