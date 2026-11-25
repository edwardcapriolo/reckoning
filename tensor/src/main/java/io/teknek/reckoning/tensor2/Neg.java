package io.teknek.reckoning.tensor2;

/** Element-wise negation into a caller-owned destination tensor. */
public class Neg {
    private final TensorRef source;
    private TensorRef destination;

    public Neg(TensorRef source) {
        this.source = source;
    }

    public Neg into(TensorRef destination) {
        this.destination = destination;
        return this;
    }

    public TensorRef getSource() {
        return source;
    }

    public TensorRef getDestination() {
        return destination;
    }
}
