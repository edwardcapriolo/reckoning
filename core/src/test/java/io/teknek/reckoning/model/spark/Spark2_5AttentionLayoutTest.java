package io.teknek.reckoning.model.spark;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Spark2_5AttentionLayoutTest {
    @Test
    void splitsFusedQkvAndRepeatsTwoKvHeadsAcrossEightQueryHeads() {
        Lighter lighter = new Lighter();
        try (TensorRef fused = lighter.allocate(DType.F32, TensorShape.of(1, 48))) {
            for (int column = 0; column < fused.shape().last(); column++) {
                fused.set(column + 1.0f, 0, column);
            }
            try (Spark2_5AttentionLayout.Split split = Spark2_5AttentionLayout.splitQkv(fused, 8, 2, 4, lighter);
                 TensorRef repeated = Spark2_5AttentionLayout.repeatKeyValue(split.key(), 8, 2, 4, lighter)) {
                assertEquals(32, split.query().shape().last());
                assertEquals(8, split.key().shape().last());
                assertEquals(8, split.value().shape().last());
                assertEquals(1.0f, split.query().get(0, 0));
                assertEquals(33.0f, split.key().get(0, 0));
                assertEquals(41.0f, split.value().get(0, 0));
                assertEquals(33.0f, repeated.get(0, 0));
                assertEquals(33.0f, repeated.get(0, 4));
                assertEquals(37.0f, repeated.get(0, 16));
                assertEquals(37.0f, repeated.get(0, 20));
            }
        }
    }
}
