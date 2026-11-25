package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Cat;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import net.jafama.FastMath;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Per-forward cosine and sine tables shared by Spark layers of the same type. */
public final class Spark2_5RopeCache implements AutoCloseable {
    private final Map<String, PositionEmbeddings> embeddings;

    private Spark2_5RopeCache(Map<String, PositionEmbeddings> embeddings) {
        this.embeddings = Map.copyOf(embeddings);
    }

    public static Spark2_5RopeCache create(Spark2_5Config config, int startPosition, int sequenceLength,
            Lighter lighter) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(lighter, "lighter");
        if (startPosition < 0 || sequenceLength < 0) {
            throw new IllegalArgumentException("Invalid Spark RoPE position range");
        }
        Map<String, PositionEmbeddings> tables = new LinkedHashMap<>();
        try {
            for (String layerType : config.layerTypes.stream().distinct().toList()) {
                int rotaryDimension = config.rotaryDimension(layerType);
                double theta = config.ropeTheta(layerType);
                tables.put(layerType, computeRopeCosSin(startPosition, sequenceLength,
                        rotaryDimension, theta, lighter));
            }
            return new Spark2_5RopeCache(tables);
        } catch (RuntimeException | Error error) {
            tables.values().forEach(PositionEmbeddings::close);
            throw error;
        }
    }

    /** Java counterpart of the source model's compute_rope_cos_sin helper. */
    static PositionEmbeddings computeRopeCosSin(int startPosition, int sequenceLength,
            int ropeHeadDim, double ropeTheta, Lighter lighter) {
        int halfRotary = ropeHeadDim / 2;
        TensorRef cos = lighter.allocate(DType.F32, TensorShape.of(sequenceLength, ropeHeadDim));
        TensorRef sin = lighter.allocate(DType.F32, TensorShape.of(sequenceLength, ropeHeadDim));
        try (TensorRef positions = lighter.allocate(DType.F32, TensorShape.of(sequenceLength, 1));
             TensorRef invFreq = lighter.allocate(DType.F32, TensorShape.of(1, halfRotary));
             TensorRef freqs = lighter.allocate(DType.F32, TensorShape.of(sequenceLength, halfRotary));
             TensorRef duplicatedFreqs = lighter.allocate(DType.F32, TensorShape.of(sequenceLength, ropeHeadDim))) {
            // torch.arange(...).to(dtype=torch.float) and positions.to(dtype=torch.float32)
            for (int row = 0; row < sequenceLength; row++) {
                positions.set(startPosition + row, row, 0);
            }
            for (int dimension = 0; dimension < halfRotary; dimension++) {
                invFreq.set((float) (1.0 / FastMath.pow(ropeTheta,
                        (2.0 * dimension) / ropeHeadDim)), 0, dimension);
            }

            // torch.outer(t, inv_freq)
            for (int row = 0; row < sequenceLength; row++) {
                for (int dimension = 0; dimension < halfRotary; dimension++) {
                    freqs.set(positions.get(row, 0) * invFreq.get(0, dimension), row, dimension);
                }
            }

            // torch.cat([freqs, freqs], dim=-1)
            lighter.cat(new Cat().result(duplicatedFreqs).inputs(freqs, freqs));

            // freqs.cos() and freqs.sin(), as separate tensor results.
            for (int row = 0; row < sequenceLength; row++) {
                for (int dimension = 0; dimension < ropeHeadDim; dimension++) {
                    float frequency = duplicatedFreqs.get(row, dimension);
                    cos.set((float) FastMath.cos(frequency), row, dimension);
                    sin.set((float) FastMath.sin(frequency), row, dimension);
                }
            }
            return new PositionEmbeddings(cos, sin);
        } catch (RuntimeException | Error error) {
            cos.close();
            sin.close();
            throw error;
        }
    }

    public PositionEmbeddings forLayerType(String layerType) {
        PositionEmbeddings result = embeddings.get(layerType);
        if (result == null) {
            throw new IllegalArgumentException("No Spark RoPE table for layer type " + layerType);
        }
        return result;
    }

    @Override
    public void close() {
        embeddings.values().forEach(PositionEmbeddings::close);
    }

    public record PositionEmbeddings(TensorRef cos, TensorRef sin) implements AutoCloseable {
        @Override
        public void close() {
            cos.close();
            sin.close();
        }
    }
}
