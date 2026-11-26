package io.teknek.reckoning.model.spark;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.EmbedInput;
import io.teknek.reckoning.generator2.TransformerBlock2;
import io.teknek.reckoning.generator2.ForwardContext;
import io.teknek.reckoning.generator.ForwardPhase;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.model.llama.LlamaModel;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.toolcallparser.ToolCallParser;
import io.teknek.reckoning.model.tensorparallel.TensorParallelCollectives;
import io.teknek.reckoning.model.tensorparallel.TensorParallelContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.function.Consumer;
import java.util.List;
import java.util.stream.IntStream;

import io.teknek.reckoning.tensor.kv.KvCacheSession;
import io.teknek.reckoning.tensorlib.PlannedTensorRef;

/** Initial Spark model shell; Spark-specific forward behavior is not implemented yet. */
public class Spark2_5Model extends LlamaModel {
    private static final Logger LOGGER = LoggerFactory.getLogger(Spark2_5Model.class);
    private Spark2_5LayerWeights[] layerWeights;

    public Spark2_5Model(InferenceType inferenceType, Config config, WeightLoader weights,
            PreTrainedTokenizer tokenizer, DType workingMemoryDType, DType workingQType,
            Optional<DType> modelQType, MetricRegistry metricRegistry,
            KvBufferCacheSettings kvBufferCacheSettings, ToolCallParser toolCallParser,
            WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization,
            Lighter lighter) {
        super(inferenceType, config, weights, tokenizer, workingMemoryDType, workingQType, modelQType,
                metricRegistry, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization, lighter);
    }

    @Override
    protected EmbedInput loadInputWeights() {
        LOGGER.info("spark load input embedding");
        embedTokenWeightsRef = registerModelTensorRef(weights.loadRef("model.embedding.weight"));


        return new EmbedInput(this) {
            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int unusedPosition) {
                try (TensorRef row = embedTokenWeightsRef.slice(inputToken)) {
                    TensorRef embedding = makeDenseTensorRef(1, config.embeddingLength);
                    if (row.dType() == embedding.dType()) {
                        lighter.copy(row, 0, embedding, 0, config.embeddingLength);
                    } else {
                        try (TensorRef converted = lighter.reshape(row, embedding.dType())) {
                            lighter.copy(converted, 0, embedding, 0, config.embeddingLength);
                        }
                    }
                    return embedding;
                }
            }
        };
    }

    @Override
    protected boolean supportsLoraHotSwap() { return false; }

    @Override
    protected boolean usesTensorRefExecution() {
        return true;
    }

    @Override
    public PlannedTensorRef forward(PlannedTensorRef embedding, int startPos, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase,
            int batchSize, int sequenceLength, int[] attentionMask, ForwardContext ignored) {
        Spark2_5Config sparkConfig = (Spark2_5Config) config;
        try (Spark2_5RopeCache cache = Spark2_5RopeCache.create(sparkConfig, startPos, sequenceLength, lighter)) {
            Spark2_5ForwardContext context = new Spark2_5ForwardContext(cache,
                    positions(startPos, sequenceLength), attentionMask);
            return super.forward(embedding, startPos, kvSession, tensorReducer, phase,
                    batchSize, sequenceLength, attentionMask, context);
        }
    }

    private static int[] positions(int startPosition, int sequenceLength) {
        int[] positions = new int[sequenceLength];
        for (int index = 0; index < sequenceLength; index++) {
            positions[index] = startPosition + index;
        }
        return positions;
    }

    @Override
    public boolean usesKvCache2Generation() {
        return true;
    }

    @Override
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        DType qType = modelQType.orElse(this.modelDType);
        layerWeights = new Spark2_5LayerWeights[config.numberOfLayers];
        Spark2_5Config sparkConfig = (Spark2_5Config) config;
        TransformerBlock2[] blocks = new TransformerBlock2[config.numberOfLayers];

        IntStream.range(0, config.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String attention = base + "self_attn.";
            String mlp = base + "mlp.";
            layerWeights[i] = new Spark2_5LayerWeights(
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            base + "input_layernorm.weight", weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            base + "post_attention_layernorm.weight", weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            attention + "q_k_v_proj.weight", weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            attention + "g_proj.weight", weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            attention + "out_proj.weight", weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            mlp + "gate_proj.weight", weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            mlp + "up_proj.weight", weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            mlp + "down_proj.weight", weights::loadRef, qType)));
            blocks[i] = Spark2_5TransformerBlock.create(this, sparkConfig, i, layerWeights[i], lighter);
        });
        return blocks;
    }

}
