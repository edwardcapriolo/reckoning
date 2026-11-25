package io.teknek.reckoning.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorDisplayUtil;
import io.teknek.reckoning.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorRefViewTest {
    @Test
    void displaysZeroCopyColumnViewsAndPreservesParentAliasing() {
        Lighter lighter = new Lighter(new MetricRegistry());
        try (TensorRef source = lighter.allocate(DType.F32, TensorShape.of(2, 16));
             TensorRef firstHalf = lighter.view(new RowView()
                     .source(source).columnOffset(0).columnLength(8));
             TensorRef secondHalf = lighter.view(new RowView()
                     .source(source).columnOffset(8).columnLength(8))) {
            for (int row = 0; row < 2; row++) {
                for (int column = 0; column < 16; column++) {
                    source.set(row * 16.0f + column, row, column);
                }
            }

            firstHalf.set(99.0f, 1, 7);
            assertEquals(99.0f, source.get(1, 7));
            TensorRefBackedTensor firstDisplay = new TensorRefBackedTensor(firstHalf);
            TensorRefBackedTensor secondDisplay = new TensorRefBackedTensor(secondHalf);
            assertEquals("[0][0]=  0.0000 [0][1]=  1.0000 [0][2]=  2.0000 [0][3]=  3.0000 "
                            + "[0][4]=  4.0000 [0][5]=  5.0000 [0][6]=  6.0000 [0][7]=  7.0000 \n"
                            + "[1][0]= 16.0000 [1][1]= 17.0000 [1][2]= 18.0000 [1][3]= 19.0000 "
                            + "[1][4]= 20.0000 [1][5]= 21.0000 [1][6]= 22.0000 [1][7]= 99.0000 \n",
                    TensorDisplayUtil.pretty2dDisplayAll(firstDisplay));
            assertEquals("[0][0]=  8.0000 [0][1]=  9.0000 [0][2]= 10.0000 [0][3]= 11.0000 "
                            + "[0][4]= 12.0000 [0][5]= 13.0000 [0][6]= 14.0000 [0][7]= 15.0000 \n"
                            + "[1][0]= 24.0000 [1][1]= 25.0000 [1][2]= 26.0000 [1][3]= 27.0000 "
                            + "[1][4]= 28.0000 [1][5]= 29.0000 [1][6]= 30.0000 [1][7]= 31.0000 \n",
                    TensorDisplayUtil.pretty2dDisplayAll(secondDisplay));
        }
    }

    @Test
    void rejectsUnsupportedViewSourcesAndRanges() {
        Lighter lighter = new Lighter(new MetricRegistry());
        try (TensorRef rankThree = lighter.allocate(DType.F32, TensorShape.of(2, 3, 4));
             TensorRef f16 = lighter.allocate(DType.F16, TensorShape.of(2, 16));
             TensorRef i8 = lighter.allocate(DType.I8, TensorShape.of(2, 32));
             TensorRef q4 = lighter.allocate(DType.Q4, TensorShape.of(2, 32));
             TensorRef source = lighter.allocate(DType.F32, TensorShape.of(2, 16))) {
            assertThrows(IllegalArgumentException.class,
                    () -> lighter.view(new RowView().source(rankThree).columnOffset(0).columnLength(2)));
            assertThrows(IllegalArgumentException.class,
                    () -> lighter.view(new RowView().source(f16).columnOffset(0).columnLength(2)));
            assertThrows(IllegalArgumentException.class,
                    () -> lighter.view(new RowView().source(i8).columnOffset(0).columnLength(2)));
            assertThrows(IllegalArgumentException.class,
                    () -> lighter.view(new RowView().source(q4).columnOffset(0).columnLength(2)));
            assertThrows(IllegalArgumentException.class,
                    () -> lighter.view(new RowView().source(source).columnOffset(-1).columnLength(2)));
            assertThrows(IllegalArgumentException.class,
                    () -> lighter.view(new RowView().source(source).columnOffset(15).columnLength(2)));
            assertThrows(IllegalArgumentException.class,
                    () -> lighter.view(new RowView().source(source).columnOffset(0).columnLength(-1)));
            assertEquals(0, lighter.view(new RowView().source(source).columnOffset(16).columnLength(0))
                    .shape().last());
        }
    }

    @Test
    void viewCannotBeCreatedFromClosedParent() {
        Lighter lighter = new Lighter(new MetricRegistry());
        TensorRef source = lighter.allocate(DType.F32, TensorShape.of(2, 16));
        source.close();
        assertThrows(RuntimeException.class,
                () -> lighter.view(new RowView().source(source).columnOffset(0).columnLength(8)));
    }

    @Test
    void viewCannotBeReadAfterParentCloses() {
        Lighter lighter = new Lighter(new MetricRegistry());
        TensorRef source = lighter.allocate(DType.F32, TensorShape.of(2, 16));
        TensorRef view = lighter.view(new RowView().source(source).columnOffset(0).columnLength(8));
        source.close();
        assertThrows(RuntimeException.class, () -> view.get(0, 0));
    }

    @Test
    void columnViewWorksAsProviderInputAcrossRows() {
        Lighter lighter = new Lighter(new MetricRegistry());
        try (TensorRef source = lighter.allocate(DType.F32, TensorShape.of(2, 16));
             TensorRef weights = lighter.allocate(DType.F32, TensorShape.of(3, 8));
             TensorRef dotOutput = lighter.allocate(DType.F32, TensorShape.of(2, 3));
             TensorRef rowOutput = lighter.allocate(DType.F32, TensorShape.of(2, 3));
             TensorRef alpha = lighter.allocate(DType.F32, TensorShape.of(1, 2));
             TensorRef saxpyOutput = lighter.allocate(DType.F32, TensorShape.of(1, 8))) {
            for (int row = 0; row < 2; row++) {
                for (int column = 0; column < 16; column++) {
                    source.set(row * 16.0f + column, row, column);
                }
            }
            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 8; column++) {
                    weights.set((row + 1) * (column + 1), row, column);
                }
            }
            try (TensorRef view = lighter.view(new RowView()
                    .source(source).columnOffset(4).columnLength(8))) {
                lighter.batchDotProduct(new BatchDotProduct()
                        .result(dotOutput).a(view).b(weights)
                        .aColumnOffset(0).bColumnOffset(0).columnLength(8)
                        .resultRowOffset(0).bRowOffset(0).rowChunkSize(3));
                lighter.dotProductRows(rowOutput, view, weights, 0, 0, 8, 0, 3, 0);
                alpha.set(0.25f, 0, 0);
                alpha.set(0.75f, 0, 1);
                lighter.saxpy(alpha, view, saxpyOutput, 0, 0, 8, 0, 0, 2);
            }
            for (int row = 0; row < 2; row++) {
                for (int weight = 0; weight < 3; weight++) {
                    float expected = 0.0f;
                    for (int column = 0; column < 8; column++) {
                        expected += source.get(row, column + 4) * weights.get(weight, column);
                    }
                    assertEquals(expected, dotOutput.get(row, weight), 1.0e-4f,
                            "row=" + row + ", weight=" + weight);
                    assertEquals(expected, rowOutput.get(row, weight), 1.0e-4f,
                            "row dot row=" + row + ", weight=" + weight);
                }
            }
            String dotDisplay = TensorDisplayUtil.pretty2dDisplayAll(new TensorRefBackedTensor(dotOutput));
            String rowDisplay = TensorDisplayUtil.pretty2dDisplayAll(new TensorRefBackedTensor(rowOutput));
            String saxpyDisplay = TensorDisplayUtil.pretty2dDisplayAll(new TensorRefBackedTensor(saxpyOutput));
            System.out.println("[view-test] batch-dot\n" + dotDisplay
                    + "[view-test] row-dot\n" + rowDisplay
                    + "[view-test] batch-saxpy\n" + saxpyDisplay);
            assertEquals(dotDisplay, rowDisplay);
            assertEquals("[0][0]= 16.0000 [0][1]= 17.0000 [0][2]= 18.0000 [0][3]= 19.0000 "
                            + "[0][4]= 20.0000 [0][5]= 21.0000 [0][6]= 22.0000 [0][7]= 23.0000 \n",
                    saxpyDisplay);
        }
    }
}
