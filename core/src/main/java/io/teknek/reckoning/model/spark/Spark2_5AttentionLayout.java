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
        // TODO(Spark): replace these copies with writable column views or a fused projection layout
        // once TensorRef providers support the required strided views. Correctness comes first here.
        Objects.requireNonNull(fused, "fused");
        Objects.requireNonNull(lighter, "lighter");
        validate(fused, queryHeads, keyValueHeads, headSize);
        int rows = fused.shape().first();
        int queryWidth = queryHeads * headSize;
        int keyValueWidth = keyValueHeads * headSize;
        TensorRef query = lighter.allocate(fused.dType(), TensorShape.of(rows, queryWidth));
        TensorRef key = lighter.allocate(fused.dType(), TensorShape.of(rows, keyValueWidth));
        TensorRef value = lighter.allocate(fused.dType(), TensorShape.of(rows, keyValueWidth));
        try {
            for (int row = 0; row < rows; row++) {
                int source = row * fused.shape().last();
                lighter.copy(fused, source, query, row * queryWidth, queryWidth);
                lighter.copy(fused, source + queryWidth, key, row * keyValueWidth, keyValueWidth);
                lighter.copy(fused, source + queryWidth + keyValueWidth, value,
                        row * keyValueWidth, keyValueWidth);
            }
            return new Split(query, key, value);
        } catch (RuntimeException | Error error) {
            query.close();
            key.close();
            value.close();
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
