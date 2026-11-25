package io.teknek.reckoning.safetensors.prompt.local;

import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LocalSparkPromptTemplateTest {
    @Test
    void rendersThinkingEffortVariants() {
        PromptSupport support = LocalPromptTemplateFixtures.promptSupport("XHToken_Spark-X2.6-1.7B");

        PromptContext high = support.builder()
                .addTemplateArgs(Map.of("enable_thinking", true, "reasoning_effort", "high"))
                .addUserMessage("What is the capital of Anhui Province?")
                .build();
        PromptContext low = support.builder()
                .addTemplateArgs(Map.of("enable_thinking", true, "reasoning_effort", "low"))
                .addUserMessage("What is the capital of Anhui Province?")
                .build();

        System.out.println("SPARK_PROMPT_HIGH_START");
        System.out.println(high.getPrompt());
        System.out.println("SPARK_PROMPT_HIGH_END");
        System.out.println("SPARK_PROMPT_LOW_START");
        System.out.println(low.getPrompt());
        System.out.println("SPARK_PROMPT_LOW_END");

        assertTrue(high.getPrompt().contains("thinking_effort=high"), high.getPrompt());
        assertTrue(low.getPrompt().contains("thinking_effort=low"), low.getPrompt());
        assertNotEquals(high.getPrompt(), low.getPrompt());
    }
}
