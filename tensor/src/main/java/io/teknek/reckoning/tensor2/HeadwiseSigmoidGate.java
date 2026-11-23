package io.teknek.reckoning.tensor2;

import java.util.Objects;

/** Applies one sigmoid gate to every head's contiguous hidden-width slice. */
public final class HeadwiseSigmoidGate {
    private final TensorRef target;
    private final TensorRef gateScores;
    private final int heads;
    private final int headSize;

    public HeadwiseSigmoidGate(TensorRef target, TensorRef gateScores, int heads, int headSize) {
        this.target = Objects.requireNonNull(target, "target");
        this.gateScores = Objects.requireNonNull(gateScores, "gateScores");
        this.heads = heads;
        this.headSize = headSize;
    }

    public TensorRef target() { return target; }
    public TensorRef gateScores() { return gateScores; }
    public int heads() { return heads; }
    public int headSize() { return headSize; }
}
