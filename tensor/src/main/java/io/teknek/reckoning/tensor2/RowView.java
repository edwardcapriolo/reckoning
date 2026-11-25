package io.teknek.reckoning.tensor2;

import java.util.Objects;

/** Describes a zero-copy rank-2 view over a contiguous column range. */
public final class RowView {
    private TensorRef source;
    private int columnOffset;
    private int columnLength;

    public RowView source(TensorRef source) {
        this.source = Objects.requireNonNull(source, "source");
        return this;
    }

    public RowView columnOffset(int columnOffset) {
        this.columnOffset = columnOffset;
        return this;
    }

    public RowView columnLength(int columnLength) {
        this.columnLength = columnLength;
        return this;
    }

    public TensorRef source() { return source; }
    public int columnOffset() { return columnOffset; }
    public int columnLength() { return columnLength; }
}
