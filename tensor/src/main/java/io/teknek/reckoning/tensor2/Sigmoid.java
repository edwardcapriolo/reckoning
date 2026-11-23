package io.teknek.reckoning.tensor2;

import java.util.Objects;

/** In-place elementwise sigmoid operation. */
public final class Sigmoid {
    private final TensorRef target;

    public Sigmoid(TensorRef target) {
        this.target = Objects.requireNonNull(target, "target");
    }

    public TensorRef target() {
        return target;
    }
}
