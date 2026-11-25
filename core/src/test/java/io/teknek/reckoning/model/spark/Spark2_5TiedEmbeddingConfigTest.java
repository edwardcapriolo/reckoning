package io.teknek.reckoning.model.spark;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Spark2_5TiedEmbeddingConfigTest {
    @Test
    void tieWordEmbeddingsIsAnExplicitConfigPolicy() {
        assertTrue(Spark2_5TestHarness.config(true).tieWordEmbeddings(),
                "true selects model.embedding.weight for output logits");
        assertFalse(Spark2_5TestHarness.config(false).tieWordEmbeddings(),
                "false requires a separate output projection weight");
    }
}
