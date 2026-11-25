package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.JsonUtils;
import io.teknek.reckoning.safetensors.DefaultWeightLoader;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.tensor.TensorInfo;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Phase 2 inventory of the exact Spark-X2.6-1.7B checkpoint. */
@Tag("large-model")
class Spark2_5InventoryIT {
    private static final String OWNER = "XHToken";
    private static final String MODEL = "Spark-X2.6-1.7B";

    @Test
    void downloadsAndInventoriesTheCompleteCheckpoint() throws Exception {
        ModelFetcher fetcher = new ModelFetcher(OWNER, MODEL).withDownload(true);
        File modelDirectory = fetcher.maybeDownload(ModelFetcher.FetchPolicy.FULL_MODEL);
        Path root = modelDirectory.toPath();

        Spark2_5Config config = JsonUtils.om.readValue(root.resolve("config.json").toFile(), Spark2_5Config.class);
        assertEquals("spark2_5", config.modelType);
        assertEquals(List.of("Spark2_5ForCausalLM"), config.architectures);
        assertEquals(1_048_576, config.contextLength);
        assertEquals(2_048, config.embeddingLength);
        assertEquals(6_656, config.hiddenLength);
        assertEquals(28, config.numberOfLayers);
        assertEquals(8, config.numberOfHeads);
        assertEquals(2, config.numberOfKeyValueHeads);
        assertEquals(256, config.headSize);
        assertEquals(512, config.slidingWindow);
        assertEquals(131_072, config.vocabularySize);
        assertEquals(64, config.rotaryDimension("full_attention"));
        assertEquals(256, config.rotaryDimension("sliding_attention"));
        assertEquals(5_000_000.0, config.ropeTheta("full_attention"));
        assertEquals(10_000.0, config.ropeTheta("sliding_attention"));

        for (String file : List.of(
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors")) {
            assertTrue(Files.isRegularFile(root.resolve(file)), "missing downloaded file " + file);
            assertTrue(Files.size(root.resolve(file)) > 0, "empty downloaded file " + file);
        }
        assertTrue(Files.exists(root.resolve(".finished")), "full fetch must create completion marker");

        try (DefaultWeightLoader weights = new DefaultWeightLoader(modelDirectory)) {
            Map<String, TensorInfo> tensors = weights.tensorInfoMap();
            assertEquals(226, tensors.size(), "embedding, norm, and 28 * 8 layer tensors");
            assertTrue(tensors.values().stream().allMatch(info -> info.dType.name().equals("BF16")),
                    "all checkpoint tensors must be BF16");

            assertShape(tensors, "model.embedding.weight", 131_072, 2_048);
            assertShape(tensors, "model.norm.weight", 2_048);
            for (int layer = 0; layer < config.numberOfLayers; layer++) {
                String prefix = "model.layers." + layer;
                assertShape(tensors, prefix + ".input_layernorm.weight", 2_048);
                assertShape(tensors, prefix + ".post_attention_layernorm.weight", 2_048);
                assertShape(tensors, prefix + ".self_attn.q_k_v_proj.weight", 3_072, 2_048);
                assertShape(tensors, prefix + ".self_attn.g_proj.weight", 8, 2_048);
                assertShape(tensors, prefix + ".self_attn.out_proj.weight", 2_048, 2_048);
                assertShape(tensors, prefix + ".mlp.gate_proj.weight", 6_656, 2_048);
                assertShape(tensors, prefix + ".mlp.up_proj.weight", 6_656, 2_048);
                assertShape(tensors, prefix + ".mlp.down_proj.weight", 2_048, 6_656);
            }
            assertFalse(tensors.containsKey("lm_head.weight"), "tied output embedding must not have a separate tensor");
        }

        try (Stream<Path> files = Files.walk(root)) {
            assertTrue(files.filter(Files::isRegularFile).count() >= 8, "full fetch must contain model artifacts");
        }
    }

    private static void assertShape(Map<String, TensorInfo> tensors, String name, int... expected) {
        TensorInfo info = tensors.get(name);
        assertNotNull(info, "missing tensor " + name);
        assertEquals(java.util.Arrays.stream(expected).boxed().toList(), java.util.Arrays.stream(info.shape).boxed().toList(),
                "shape for " + name);
    }
}
