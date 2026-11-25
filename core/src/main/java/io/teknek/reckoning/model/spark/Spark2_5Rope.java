package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Cat;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.Neg;
import io.teknek.reckoning.tensor2.RowView;
import io.teknek.reckoning.tensor2.TensorRef;
import net.jafama.FastMath;

import java.util.Objects;

/** Layer-type-specific, on-demand RoPE for Spark attention heads. */
public final class Spark2_5Rope implements AutoCloseable {
    private final int headSize;
    private final int rotaryDimension;
    private final TensorRef inverseFrequencies;
    private final Lighter lighter;

    public Spark2_5Rope(Spark2_5Config config, String layerType, Lighter lighter) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(layerType, "layerType");
        Objects.requireNonNull(lighter, "lighter");
        this.headSize = config.headSize;
        int rotaryDimension = config.rotaryDimension(layerType);
        if (rotaryDimension <= 0 || (rotaryDimension & 1) != 0 || rotaryDimension > headSize) {
            throw new IllegalArgumentException("Invalid Spark rotary dimension: " + rotaryDimension);
        }
        int halfHead = headSize / 2;
        int halfRotary = rotaryDimension / 2;
        this.inverseFrequencies = lighter.allocate(DType.F32, TensorShape.of(1, halfHead));
        try {
            double theta = config.ropeTheta(layerType);
            for (int index = 0; index < halfRotary; index++) {
                inverseFrequencies.set((float) (1.0 / FastMath.pow(theta,
                        (2.0 * index) / rotaryDimension)), 0, index);
            }
            for (int index = halfRotary; index < halfHead; index++) {
                inverseFrequencies.set(0.0f, 0, index);
            }
        } catch (RuntimeException | Error error) {
            inverseFrequencies.close();
            throw error;
        }
        this.lighter = lighter;
        this.rotaryDimension = rotaryDimension;
    }

    /** Faithful rank-2 equivalent of the source model's {@code rotate_half} helper. */
    public TensorRef rotateHalf(TensorRef source) {
        Objects.requireNonNull(source, "source");
        if (source.dims() != 2 || source.shape().last() % 2 != 0
                || (source.dType() != DType.F32 && source.dType() != DType.BF16)) {
            throw new IllegalArgumentException("rotateHalf requires an even rank-2 F32 or BF16 tensor");
        }
        int half = source.shape().last() / 2;
        TensorRef result = lighter.allocate(source.dType(),
                TensorShape.of(source.shape().first(), source.shape().last()));
        try (TensorRef first = lighter.view(new RowView()
                     .source(source).columnOffset(0).columnLength(half));
             TensorRef second = lighter.view(new RowView()
                     .source(source).columnOffset(half).columnLength(half));
             TensorRef negatedSecond = lighter.allocate(source.dType(),
                     TensorShape.of(source.shape().first(), half))) {
            lighter.neg(new Neg(second).into(negatedSecond));
            lighter.cat(new Cat().result(result).inputs(negatedSecond, first));
            return result;
        } catch (RuntimeException | Error failure) {
            result.close();
            throw failure;
        }
    }

    /** Applies this layer type's rotation in place without materializing a position table. */
    public void apply(TensorRef target, int heads, int startPosition) {
        Objects.requireNonNull(target, "target");
        if (target.dims() != 2 || heads < 1 || startPosition < 0
                || (headSize & 1) != 0 || target.shape().last() < heads * headSize) {
            throw new IllegalArgumentException("Invalid Spark RoPE target shape or position");
        }
        applyRotationsUsingRotateHalf(target, heads, startPosition);
    }

    public void apply(TensorRef target, int heads, int startPosition,
            Spark2_5RopeCache.PositionEmbeddings embeddings) {
        Objects.requireNonNull(embeddings, "embeddings");
        Objects.requireNonNull(target, "target");
        if (target.dims() != 2 || heads < 1 || startPosition < 0
                || target.shape().last() < heads * headSize
                || embeddings.cos().shape().first() < target.shape().first()
                || embeddings.cos().shape().last() != rotaryDimension) {
            throw new IllegalArgumentException("Invalid Spark RoPE target or embeddings");
        }
        applyRotationsUsingTables(target, heads, embeddings);
    }

    private void applyRotationsUsingRotateHalf(TensorRef target, int heads, int startPosition) {
        int halfRotary = rotaryDimension / 2;
        for (int head = 0; head < heads; head++) {
            int offset = head * headSize;
            try (TensorRef rotaryView = lighter.view(new RowView()
                    .source(target).columnOffset(offset).columnLength(rotaryDimension));
                 TensorRef rotated = rotateHalf(rotaryView)) {
                for (int row = 0; row < target.shape().first(); row++) {
                    int position = startPosition + row;
                    for (int dimension = 0; dimension < halfRotary; dimension++) {
                        double angle = position * (double) inverseFrequencies.get(0, dimension);
                        float cosine = (float) FastMath.cos(angle);
                        float sine = (float) FastMath.sin(angle);
                        float x0 = target.get(row, offset + dimension);
                        float x1 = target.get(row, offset + dimension + halfRotary);
                        float rotated0 = rotated.get(row, dimension);
                        float rotated1 = rotated.get(row, dimension + halfRotary);
                        target.set(x0 * cosine + rotated0 * sine, row, offset + dimension);
                        target.set(x1 * cosine + rotated1 * sine,
                                row, offset + dimension + halfRotary);
                    }
                }
            }
        }
    }

    private void applyRotationsUsingTables(TensorRef target, int heads,
            Spark2_5RopeCache.PositionEmbeddings embeddings) {
        int halfRotary = rotaryDimension / 2;
        for (int head = 0; head < heads; head++) {
            int offset = head * headSize;
            try (TensorRef rotaryView = lighter.view(new RowView()
                    .source(target).columnOffset(offset).columnLength(rotaryDimension))) {
                TensorRef working = target.dType() == DType.F32
                        ? rotaryView : lighter.reshape(rotaryView, DType.F32);
                TensorRef result = lighter.allocate(DType.F32,
                        TensorShape.of(target.shape().first(), rotaryDimension));
                try (TensorRef rotated = rotateHalf(working)) {
                for (int row = 0; row < target.shape().first(); row++) {
                    for (int dimension = 0; dimension < halfRotary; dimension++) {
                        float cosine = embeddings.cos().get(row, dimension);
                        float sine = embeddings.sin().get(row, dimension);
                        float x0 = working.get(row, dimension);
                        float x1 = working.get(row, dimension + halfRotary);
                        float rotated0 = rotated.get(row, dimension);
                        float rotated1 = rotated.get(row, dimension + halfRotary);
                        result.set(x0 * cosine + rotated0 * sine, row, dimension);
                        result.set(x1 * cosine + rotated1 * sine, row, dimension + halfRotary);
                    }
                }
                TensorRef converted = target.dType() == DType.F32
                        ? result : lighter.reshape(result, target.dType());
                try {
                    for (int row = 0; row < target.shape().first(); row++) {
                        int rowOffset = row * rotaryDimension;
                        lighter.copy(converted, rowOffset, rotaryView, rowOffset, rotaryDimension);
                    }
                } finally {
                    if (converted != result) converted.close();
                }
                } finally {
                    result.close();
                    if (working != rotaryView) working.close();
                }
            }
        }
    }

    @Override
    public void close() {
        inverseFrequencies.close();
    }
}
