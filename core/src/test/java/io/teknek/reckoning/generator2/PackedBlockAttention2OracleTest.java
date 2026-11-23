package io.teknek.reckoning.generator2;

import io.dropwizard.metrics5.MetricRegistry;
import org.apache.commons.math3.util.FastMath;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.math.BiIntConsumer;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.CompositeOps;
import io.teknek.reckoning.tensor2.BatchDotProduct;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Scalar oracle for the packed attention mechanics, independent of tensor kernels. */
class PackedBlockAttention2OracleTest {
    @Test
    void matchesScalarGqaCausalAndSlidingAttention() {
        int queryRows = 3;
        int prefixRows = 2;
        int heads = 4;
        int kvHeads = 2;
        int headSize = 1;
        float scale = 0.7f;
        Float softcap = null;
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        CompositeOps compositeOps = new CompositeOps(lighter, metrics);
        AbstractModel model = Mockito.mock(AbstractModel.class);
        Mockito.when(model.getMetricRegistry()).thenReturn(metrics);
        Mockito.when(model.getLighter()).thenReturn(lighter);
        Mockito.when(model.getCompositeOps()).thenReturn(compositeOps);
        Mockito.when(model.getPool()).thenReturn(new WrappedForkJoinPool(new ForkJoinPool(1)));
        Mockito.when(model.makeDenseTensorRef(Mockito.anyInt(), Mockito.anyInt()))
                .thenAnswer(invocation -> lighter.allocate(DType.F32,
                        TensorShape.of(invocation.getArgument(0), invocation.getArgument(1))));
        Mockito.doAnswer(invocation -> {
            BiIntConsumer action = invocation.getArgument(4);
            action.accept(invocation.getArgument(1), invocation.getArgument(2));
            return null;
        }).when(model).runChunks(Mockito.anyString(), Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt(),
                Mockito.any(BiIntConsumer.class));

        int totalRows = prefixRows + queryRows;
        int kvWidth = kvHeads * headSize;
        try (TensorRef query = lighter.allocate(DType.F32, TensorShape.of(queryRows, heads * headSize));
             TensorRef keys = lighter.allocate(DType.F32, TensorShape.of(totalRows, kvWidth));
             TensorRef values = lighter.allocate(DType.F32, TensorShape.of(totalRows, kvWidth));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(queryRows, heads * headSize))) {
            fill(query, (row, column) -> 0.13f + row * 0.41f - column * 0.07f);
            fill(keys, (row, column) -> -0.21f + row * 0.19f + column * 0.11f);
            fill(values, (row, column) -> 0.31f - row * 0.23f + column * 0.17f);

            try (TensorRef alpha = lighter.allocate(DType.F32, TensorShape.of(1, totalRows));
                 TensorRef saxpyOutput = lighter.allocate(DType.F32, TensorShape.of(1, kvWidth))) {
                fill(alpha, (row, column) -> column + 1.0f);
                lighter.saxpy(alpha, values, saxpyOutput, 0, 0, kvWidth, 1, 1, 3);
                assertEquals(2.0f * values.get(1, 0) + 3.0f * values.get(2, 0)
                        + 4.0f * values.get(3, 0), saxpyOutput.get(0, 0), 1.0e-5f);
            }

            try (TensorRef scores = lighter.allocate(DType.F32, TensorShape.of(1, totalRows))) {
                lighter.batchDotProduct(new BatchDotProduct().result(scores).a(query).b(keys)
                        .aRowOffset(1).aColumnOffset(0).bColumnOffset(0).columnLength(headSize)
                        .resultRowOffset(0).bRowOffset(0).rowChunkSize(totalRows));
                for (int position = 0; position < totalRows; position++) {
                    assertEquals(score(query, keys, 1, 0, 0, headSize, position), scores.get(0, position),
                            1.0e-5f, "score position=" + position);
                }
                compositeOps.scaledSoftMax(new io.teknek.reckoning.tensor2.ScaledSoftMax(scale)
                        .target(scores).offsetAndLength(1, 3).softcap(softcap));
                float max = Float.NEGATIVE_INFINITY;
                for (int position = 1; position < 4; position++) {
                    max = Math.max(max, score(query, keys, 1, 0, 0, headSize, position) * scale);
                }
                float denominator = 0.0f;
                for (int position = 1; position < 4; position++) {
                    denominator += (float) Math.exp(score(query, keys, 1, 0, 0, headSize, position) * scale - max);
                }
                for (int position = 1; position < 4; position++) {
                    assertEquals((float) Math.exp(score(query, keys, 1, 0, 0, headSize, position) * scale - max)
                                    / denominator,
                            scores.get(0, position), 1.0e-5f, "probability position=" + position);
                }
            }

            PackedBlockAttention2 attention = new PackedBlockAttention2(model, metrics);
            attention.forward(output, query, keys, values, prefixRows, queryRows, heads, kvHeads, headSize,
                    scale, softcap, true, 3);

            for (int row = 0; row < queryRows; row++) {
                int visibleEnd = prefixRows + row + 1;
                int windowStart = Math.max(0, visibleEnd - 3);
                for (int head = 0; head < heads; head++) {
                    int kvHead = head / (heads / kvHeads);
                    for (int component = 0; component < headSize; component++) {
                        float expected = oracle(query, keys, values, row, head, kvHead, component,
                                headSize, visibleEnd, windowStart, scale, softcap);
                        assertEquals(expected, output.get(row, head * headSize + component), 2.0e-4,
                                "row=" + row + ", head=" + head + ", component=" + component);
                    }
                }
            }
        }
    }

    private static float oracle(TensorRef query, TensorRef keys, TensorRef values, int queryRow, int head,
            int kvHead, int component, int headSize, int visibleEnd, int windowStart, float scale, Float softcap) {
        float max = Float.NEGATIVE_INFINITY;
        for (int position = windowStart; position < visibleEnd; position++) {
            float scaled = score(query, keys, queryRow, head, kvHead, headSize, position) * scale;
            max = Math.max(max, softcap == null ? scaled : (float) FastMath.tanh(scaled / softcap) * softcap);
        }
        float denominator = 0.0f;
        float result = 0.0f;
        for (int position = windowStart; position < visibleEnd; position++) {
            float scaled = score(query, keys, queryRow, head, kvHead, headSize, position) * scale;
            float transformed = softcap == null ? scaled : (float) FastMath.tanh(scaled / softcap) * softcap;
            float weight = (float) Math.exp(transformed - max);
            denominator += weight;
            result += weight * values.get(position, kvHead * headSize + component);
        }
        return result / denominator;
    }

    private static float score(TensorRef query, TensorRef keys, int queryRow, int head, int kvHead, int headSize,
            int keyRow) {
        float result = 0.0f;
        for (int component = 0; component < headSize; component++) {
            result += query.get(queryRow, head * headSize + component)
                    * keys.get(keyRow, kvHead * headSize + component);
        }
        return result;
    }

    private static void fill(TensorRef tensor, ValueFunction function) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(function.value(row, column), row, column);
            }
        }
    }

    @FunctionalInterface
    private interface ValueFunction {
        float value(int row, int column);
    }
}
