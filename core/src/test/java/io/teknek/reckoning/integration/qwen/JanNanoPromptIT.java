package io.teknek.reckoning.integration.qwen;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.model.DoNothingGenerateEvent;
import io.teknek.reckoning.model.InferenceProfiler;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.safetensors.prompt.llmon.LlmOnPrompt;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.time.Duration;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** End-to-end Jan-nano query using a simple prompt. */
@Tag("large-model")
class JanNanoPromptIT {

    @Disabled("Use this to quantize the model")
    //@Test
    void quantizesCheckpointOnDemandAndRunsStructuredLlmonQuery() throws Exception {
        ModelFetcher fetcher = new ModelFetcher("Menlo", "Jan-nano")
                .withDownload(true);
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        try {
            try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
                    .withQuantizeOnDemand(DType.Q4, "edwardcapriolo",
                            "Jan-nano")
                    .buildLocalTransformerModel()) {
                PromptContext pc =model.promptSupport().orElseThrow().builder()
                        .addUserMessage("Count from 1 to 3").build();
                Response response = model.generate(UUID.randomUUID(), pc,
                        new GeneratorParameters()
                                .withTemperature(0.0f)
                                .withSeed(42)
                                .withMaxTokens(50)
                                .withTimeout(Duration.ofSeconds(90)),
                        (a,b,c,d) -> { System.out.println(b);});
                assertTrue(response.generatedTokens != null && !response.generatedTokens.isEmpty());
                assertTrue(response.responseTextWithSpecialTokens != null
                        && !response.responseTextWithSpecialTokens.isBlank());
            }
        } finally {
            InferenceProfiler.printSummary("jano", 40);
            InferenceProfiler.printCounters();
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }
}