package io.teknek.reckoning.integration.spark;

import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.model.DeadReckoningParams;
import io.teknek.reckoning.model.DeadReckoningSelector;
import io.teknek.reckoning.model.LocalGenerationBackend;
import io.teknek.reckoning.model.RopeStorageMode;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.time.Duration;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** Java Dead Reckoning trace and timing run for the Spark-X2.6 checkpoint. */
@Tag("large-model")
class SparkDeadReckoningIT {
    @Test
    void capturesSparkTraceAndTiming() throws Exception {
        ModelFetcher fetcher = new ModelFetcher("XHToken", "Spark-X2.6-1.7B")
                .withDownload(false);
        long loadStart = System.nanoTime();
        System.out.println("[spark-progress] model-load-start");
        fetcher.maybeDownload();

        Path output = Path.of("target", "dead-reckoning", "spark-x2.6-1.7b.jsonl");
        DeadReckoningSelector selector = DeadReckoningSelector.regex(null, null, null, null,
                DeadReckoningSelector.Capture.STATS, 32,
                EnumSet.allOf(DeadReckoningSelector.Statistic.class));

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
                .withDownload(false)
                .withRopeStorageMode(RopeStorageMode.ON_DEMAND)
                .withDeadReckoning(new DeadReckoningParams(output, 32, List.of(selector)))
                .buildLocalTransformerModel()) {
            System.out.printf("[spark-progress] model-load-complete elapsed=%.1fs%n",
                    (System.nanoTime() - loadStart) / 1_000_000_000.0);
            PromptContext prompt = model.promptSupport()
                    .map(support -> support.builder()
                            .addTemplateArgs(Map.of("enable_thinking", true, "reasoning_effort", "high"))
                            .addUserMessage("What is the capital of Anhui Province?")
                            .build())
                    .orElseGet(() -> PromptContext.of("What is the capital of Anhui Province?"));
            System.out.println("[spark-progress] generation-start");
            Response response;
            try (AutoCloseable ignored = AbstractModel.withPrefillProgressListener(event ->
                    System.out.printf("[spark-progress] prefill tokens=%d/%d chunk=%d-%d layers=%d/%d elapsed=%.1fs rate=%.1f tok/s%n",
                            event.estimatedProcessedTokens(), event.totalTokens(), event.chunkStartPosition(),
                            event.chunkEndPosition(), event.processedLayers(), event.totalLayers(),
                            event.elapsedSeconds(), event.tokensPerSecond()))) {
                response = model.generateWithBackendRef(UUID.randomUUID(), prompt,
                        new GeneratorParameters()
                                .withTemperature(0.0f)
                                .withSeed(42)
                                .withMaxTokens(6)
                                .withLogProbs(true)
                                .withTopLogProbs(5)
                                .withTimeout(Duration.ofMinutes(10)),
                        (next, nextRaw, nextCleaned, timing) -> {
                            System.out.printf("[spark-progress] decode token=%d text=%s timing=%s%n",
                                    next, nextCleaned.replace("\n", "\\n"), timing);
                            System.out.flush();
                        }, new LocalGenerationBackend(model));
            }
            System.out.println("DEAD_RECKONING_RESPONSE model=Spark-X2.6-1.7B text="
                    + response.responseTextWithSpecialTokens.replace("\n", "\\n")
                    + " tokens=" + response.generatedTokens + " finish=" + response.finishReason);
            response.samplerReturns.forEach((sample) -> sample.getTopNLogProbs()
                    .ifPresent(top -> System.out.println("[spark-progress] top-logits " + top)));
        }
        System.out.println("DEAD_RECKONING_TRACE=" + output.toAbsolutePath());
    }
}
