package io.teknek.reckoning.integration.spark;

import io.teknek.reckoning.integration.qwen.AbstractQwen3BenchmarkCasesIT;
import org.junit.jupiter.api.Tag;

@Tag("large-model")
class SparkJQBenchmarkCasesIT extends AbstractQwen3BenchmarkCasesIT {
    @Override
    protected String owner() {
        return "edwardcapriolo";
    }

    @Override
    protected String modelName() {
        return "Spark-X2.6-1.7B-JQ4";
    }
}
