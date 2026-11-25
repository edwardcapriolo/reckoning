package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;

import java.util.Objects;

/** Tensor layout operations for Spark's fused QKV projection and GQA heads. */
public final class Spark2_5AttentionLayout {
    private Spark2_5AttentionLayout() {
    }

    public static Split splitQkv(TensorRef fused, int queryHeads, int keyValueHeads, int headSize,
            Lighter lighter) {
        Objects.requireNonNull(fused, "fused");
        Objects.requireNonNull(lighter, "lighter");
        validate(fused, queryHeads, keyValueHeads, headSize);
        int queryWidth = queryHeads * headSize;
        int keyValueWidth = keyValueHeads * headSize;
        TensorRef query = null;
        TensorRef key = null;
        TensorRef value = null;
        try {
            query = lighter.view(new io.teknek.reckoning.tensor2.RowView()
                    .source(fused).columnOffset(0).columnLength(queryWidth));
            key = lighter.view(new io.teknek.reckoning.tensor2.RowView()
                    .source(fused).columnOffset(queryWidth).columnLength(keyValueWidth));
            value = lighter.view(new io.teknek.reckoning.tensor2.RowView()
                    .source(fused).columnOffset(queryWidth + keyValueWidth).columnLength(keyValueWidth));
            return new Split(query, key, value);
        } catch (RuntimeException | Error error) {
            if (query != null) query.close();
            if (key != null) key.close();
            if (value != null) value.close();
            throw error;
        }
    }

    public static TensorRef repeatKeyValue(TensorRef keyValue, int queryHeads, int keyValueHeads,
            int headSize, Lighter lighter) {
        Objects.requireNonNull(keyValue, "keyValue");
        Objects.requireNonNull(lighter, "lighter");
        if (keyValue.dims() != 2 || queryHeads < 1 || keyValueHeads < 1
                || queryHeads % keyValueHeads != 0
                || keyValue.shape().last() != keyValueHeads * headSize) {
            throw new IllegalArgumentException("Invalid Spark GQA layout");
        }
        int rows = keyValue.shape().first();
        int groupSize = queryHeads / keyValueHeads;
        TensorRef result = lighter.allocate(keyValue.dType(), TensorShape.of(rows, queryHeads * headSize));
        try {
            for (int row = 0; row < rows; row++) {
                for (int kvHead = 0; kvHead < keyValueHeads; kvHead++) {
                    int source = row * keyValue.shape().last() + kvHead * headSize;
                    for (int repeat = 0; repeat < groupSize; repeat++) {
                        int targetHead = kvHead * groupSize + repeat;
                        lighter.copy(keyValue, source, result,
                                row * queryHeads * headSize + targetHead * headSize, headSize);
                    }
                }
            }
            return result;
        } catch (RuntimeException | Error error) {
            result.close();
            throw error;
        }
    }

    private static void validate(TensorRef fused, int queryHeads, int keyValueHeads, int headSize) {
        if (fused.dims() != 2 || queryHeads < 1 || keyValueHeads < 1 || headSize < 1
                || queryHeads % keyValueHeads != 0
                || fused.shape().last() != (queryHeads + 2 * keyValueHeads) * headSize) {
            throw new IllegalArgumentException("Invalid Spark fused QKV layout");
        }
    }

    public record Split(TensorRef query, TensorRef key, TensorRef value) implements AutoCloseable {
        @Override
        public void close() {
            query.close();
            key.close();
            value.close();
        }
    }
}
