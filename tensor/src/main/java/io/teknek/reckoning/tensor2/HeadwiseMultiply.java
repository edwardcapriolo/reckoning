package io.teknek.reckoning.tensor2;

import java.util.Objects;

/** Multiplies each contiguous head slice by its corresponding gate score. */
public final class HeadwiseMultiply {
    private final TensorRef target;
    private final TensorRef gates;
    private final int heads;
    private final int headSize;

    public HeadwiseMultiply(TensorRef target, TensorRef gates, int heads, int headSize) {
        this.target = Objects.requireNonNull(target, "target");
        this.gates = Objects.requireNonNull(gates, "gates");
        this.heads = heads;
        this.headSize = headSize;
    }

    public TensorRef target() { return target; }
    public TensorRef gates() { return gates; }
    public int heads() { return heads; }
    public int headSize() { return headSize; }
}
