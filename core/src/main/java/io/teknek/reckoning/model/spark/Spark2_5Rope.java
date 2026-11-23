package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.CompositeOps;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import net.jafama.FastMath;

import java.util.Objects;

/** Layer-type-specific, on-demand RoPE for Spark attention heads. */
public final class Spark2_5Rope implements AutoCloseable {
    private final int headSize;
    private final int rotaryDimension;
    private final TensorRef inverseFrequencies;
    private final CompositeOps compositeOps;

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
        this.compositeOps = new CompositeOps(lighter);
        this.rotaryDimension = rotaryDimension;
    }

    /** Applies this layer type's rotation in place without materializing a position table. */
    public void apply(TensorRef target, int heads, int startPosition) {
        Objects.requireNonNull(target, "target");
        if (target.dims() != 2 || heads < 1 || startPosition < 0
                || (headSize & 1) != 0 || target.shape().last() < heads * headSize) {
            throw new IllegalArgumentException("Invalid Spark RoPE target shape or position");
        }
        if (rotaryDimension == headSize) {
            compositeOps.rotaryEmbeddingOnDemand(target, heads, headSize, startPosition, inverseFrequencies);
            return;
        }
        int halfRotary = rotaryDimension / 2;
        for (int row = 0; row < target.shape().first(); row++) {
            int position = startPosition + row;
            for (int head = 0; head < heads; head++) {
                int offset = head * headSize;
                for (int dimension = 0; dimension < halfRotary; dimension++) {
                    float x0 = target.get(row, offset + dimension);
                    float x1 = target.get(row, offset + dimension + halfRotary);
                    double angle = position * (double) inverseFrequencies.get(0, dimension);
                    float cosine = (float) FastMath.cos(angle);
                    float sine = (float) FastMath.sin(angle);
                    target.set(x0 * cosine - x1 * sine, row, offset + dimension);
                    target.set(x0 * sine + x1 * cosine, row, offset + dimension + halfRotary);
                }
            }
        }
    }

    @Override
    public void close() {
        inverseFrequencies.close();
    }
}
