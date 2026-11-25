package io.teknek.reckoning.tensor2;

import java.util.List;
import java.util.Objects;

/** Describes materializing a rank-2 column concatenation into a caller-owned result. */
public final class Cat {
    private TensorRef result;
    private List<TensorRef> inputs;
    private int dimension = 1;

    public Cat result(TensorRef result) {
        this.result = Objects.requireNonNull(result, "result");
        return this;
    }

    public Cat inputs(List<TensorRef> inputs) {
        this.inputs = List.copyOf(Objects.requireNonNull(inputs, "inputs"));
        return this;
    }

    public Cat inputs(TensorRef... inputs) {
        return inputs(List.of(inputs));
    }

    public Cat dimension(int dimension) {
        this.dimension = dimension;
        return this;
    }

    public TensorRef result() { return result; }
    public List<TensorRef> inputs() { return inputs; }
    public int dimension() { return dimension; }
}
