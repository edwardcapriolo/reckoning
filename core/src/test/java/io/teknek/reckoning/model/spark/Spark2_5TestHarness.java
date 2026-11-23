package io.teknek.reckoning.model.spark;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.math.BiIntConsumer;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.CompositeOps;
import io.teknek.reckoning.tensor2.Lighter;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

/** Shared mocked-model wiring for small Spark characterization fixtures. */
final class Spark2_5TestHarness {
    private Spark2_5TestHarness() {
    }

    static Fixture model(Spark2_5Config config) {
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        AbstractModel model = Mockito.mock(AbstractModel.class);
        Mockito.when(model.getConfig()).thenReturn(config);
        Mockito.when(model.getMetricRegistry()).thenReturn(metrics);
        Mockito.when(model.getWorkingQType()).thenReturn(DType.F32);
        Mockito.when(model.getTensorParallelContext()).thenReturn(new StaticTensorParallelContext(0, 1));
        Mockito.when(model.getCompositeOps()).thenReturn(new CompositeOps(lighter, metrics));
        Mockito.when(model.makeDenseTensorRef(Mockito.anyInt(), Mockito.anyInt()))
                .thenAnswer(invocation -> lighter.allocate(DType.F32,
                        TensorShape.of(invocation.getArgument(0), invocation.getArgument(1))));
        Mockito.doAnswer(invocation -> {
            BiIntConsumer action = invocation.getArgument(4);
            action.accept(invocation.getArgument(1), invocation.getArgument(2));
            return null;
        }).when(model).runChunks(Mockito.anyString(), Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt(),
                Mockito.any(BiIntConsumer.class));
        return new Fixture(model, metrics, lighter);
    }

    static Spark2_5Config config(boolean tieWordEmbeddings) {
        return new Spark2_5Config(
                List.of("Spark2_5ForCausalLM"), "spark2_5", Map.of(), "float32",
                false, 0.0f, 0, 1, "sigmoid", 2, false, "gelu", 2, 0.0221f, 1,
                List.of("full_attention"), 8, false, 1, 1, 1, 2, 1.0e-6f,
                Map.of(
                        "sliding_attention", Map.of("rope_theta", 10_000.0, "partial_rotary_factor", 1.0),
                        "full_attention", Map.of("rope_theta", 5_000_000.0, "partial_rotary_factor", 0.25)),
                512, tieWordEmbeddings, true, 16);
    }

    static Spark2_5Config realShapeConfig() {
        return new Spark2_5Config(
                List.of("Spark2_5ForCausalLM"), "spark2_5", Map.of(), "bfloat16",
                false, 0.0f, 0, 1, "sigmoid", 256, true, "gelu", 2048, 0.0221f, 6656,
                List.of("sliding_attention", "sliding_attention", "sliding_attention", "full_attention"),
                1024, false, 8, 4, 2, 2, 1.0e-6f,
                Map.of(
                        "sliding_attention", Map.of("rope_theta", 10_000.0, "partial_rotary_factor", 1.0),
                        "full_attention", Map.of("rope_theta", 5_000_000.0, "partial_rotary_factor", 0.25)),
                512, true, true, 131_072);
    }

    record Fixture(AbstractModel model, MetricRegistry metrics, Lighter lighter) {
    }
}
