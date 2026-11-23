package io.teknek.reckoning.model.spark;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.reckoning.math.ActivationFunction;
import io.teknek.reckoning.safetensors.Config;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Typed configuration for the Spark 2.5 decoder architecture. */
public final class Spark2_5Config extends Config {
    public final List<String> architectures;
    public final String modelType;
    public final Map<String, String> autoMap;
    public final String dtype;
    public final boolean attentionBias;
    public final float attentionDropout;
    public final float initializerRange;
    public final boolean mlpBias;
    public final boolean tieWordEmbeddings;
    public final boolean useCache;
    public final boolean headwiseAttentionOutputGate;
    public final String gateAttentionActivationMode;
    public final Map<String, Map<String, Object>> ropeParameters;
    public final List<String> layerTypes;

    @JsonCreator
    public Spark2_5Config(
            @JsonProperty("architectures") List<String> architectures,
            @JsonProperty("model_type") String modelType,
            @JsonProperty("auto_map") Map<String, String> autoMap,
            @JsonProperty("dtype") String dtype,
            @JsonProperty("attention_bias") boolean attentionBias,
            @JsonProperty("attention_dropout") float attentionDropout,
            @JsonProperty("bos_token_id") int bosToken,
            @JsonProperty("eos_token_id") int eosToken,
            @JsonProperty("gate_attn_act_mode") String gateAttentionActivationMode,
            @JsonProperty("head_dim") int headDim,
            @JsonProperty("headwise_attn_output_gate") boolean headwiseAttentionOutputGate,
            @JsonProperty("hidden_act") String hiddenAct,
            @JsonProperty("hidden_size") int hiddenSize,
            @JsonProperty("initializer_range") float initializerRange,
            @JsonProperty("intermediate_size") int intermediateSize,
            @JsonProperty("layer_types") List<String> layerTypes,
            @JsonProperty("max_position_embeddings") int maxPositionEmbeddings,
            @JsonProperty("mlp_bias") boolean mlpBias,
            @JsonProperty("num_attention_heads") int numAttentionHeads,
            @JsonProperty("num_hidden_layers") int numHiddenLayers,
            @JsonProperty("num_key_value_heads") int numKeyValueHeads,
            @JsonProperty("pad_token_id") int padToken,
            @JsonProperty("rms_norm_eps") float rmsNormEps,
            @JsonProperty("rope_parameters") Map<String, Map<String, Object>> ropeParameters,
            @JsonProperty("sliding_window") Integer slidingWindow,
            @JsonProperty("tie_word_embeddings") boolean tieWordEmbeddings,
            @JsonProperty("use_cache") boolean useCache,
            @JsonProperty("vocab_size") int vocabularySize
    ) {
        super(
                maxPositionEmbeddings,
                hiddenSize,
                intermediateSize,
                numAttentionHeads,
                numKeyValueHeads,
                numHiddenLayers,
                rmsNormEps,
                vocabularySize,
                bosToken,
                List.of(eosToken),
                ActivationFunction.Type.valueOf(hiddenAct.toUpperCase()),
                ropeTheta(ropeParameters, "sliding_attention"),
                null,
                headDim,
                null,
                null
        );
        if (numAttentionHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException("num_attention_heads must be divisible by num_key_value_heads");
        }
        if (layerTypes == null || layerTypes.size() != numHiddenLayers) {
            throw new IllegalArgumentException("layer_types must contain one entry per hidden layer");
        }
        if (ropeParameters == null || !ropeParameters.containsKey("sliding_attention")
                || !ropeParameters.containsKey("full_attention")) {
            throw new IllegalArgumentException("Spark config requires both layer-specific RoPE parameter sets");
        }
        this.architectures = List.copyOf(architectures == null ? List.of() : architectures);
        this.modelType = modelType;
        this.autoMap = Map.copyOf(autoMap == null ? Map.of() : autoMap);
        this.dtype = dtype;
        this.attentionBias = attentionBias;
        this.attentionDropout = attentionDropout;
        this.initializerRange = initializerRange;
        this.mlpBias = mlpBias;
        this.tieWordEmbeddings = tieWordEmbeddings;
        this.useCache = useCache;
        this.headwiseAttentionOutputGate = headwiseAttentionOutputGate;
        this.gateAttentionActivationMode = gateAttentionActivationMode;
        this.ropeParameters = immutableNestedMap(ropeParameters);
        this.layerTypes = List.copyOf(layerTypes);
        this.slidingWindow = slidingWindow;
    }

    public int rotaryDimension(String layerType) {
        Map<String, Object> params = ropeParameters.get(layerType);
        if (params == null) {
            throw new IllegalArgumentException("Unknown Spark layer type: " + layerType);
        }
        Number factor = (Number) params.getOrDefault("partial_rotary_factor", 1.0);
        return (int) (headSize * factor.doubleValue());
    }

    public double ropeTheta(String layerType) {
        return ropeTheta(ropeParameters, layerType);
    }

    @Override
    public boolean tieWordEmbeddings() {
        return tieWordEmbeddings;
    }

    private static double ropeTheta(Map<String, Map<String, Object>> parameters, String layerType) {
        if (parameters == null || parameters.get(layerType) == null) {
            return 10000.0;
        }
        return ((Number) parameters.get(layerType).getOrDefault("rope_theta", 10000.0)).doubleValue();
    }

    private static Map<String, Map<String, Object>> immutableNestedMap(
            Map<String, Map<String, Object>> source) {
        Map<String, Map<String, Object>> result = new LinkedHashMap<>();
        source.forEach((key, value) -> result.put(key, Collections.unmodifiableMap(new LinkedHashMap<>(value))));
        return Collections.unmodifiableMap(result);
    }
}
