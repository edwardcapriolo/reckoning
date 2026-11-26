package io.teknek.reckoning.integration.spark;

import io.teknek.reckoning.integration.qwen.AbstractQwen3BenchmarkCasesIT;
import org.junit.jupiter.api.Tag;

@Tag("large-model")
class SparkBenchmarkCasesIT extends AbstractQwen3BenchmarkCasesIT {
    @Override
    protected String owner() {
        return "XHToken";
    }

    @Override
    protected String modelName() {
        return "Spark-X2.6-1.7B";
    }
}
