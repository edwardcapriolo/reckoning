package io.teknek.reckoning.tensor.kv;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;

class KvCacheGroupPolicyTest {
    @Test
    void fullGroupRetainsAllPositionsAndDefaultBehaviorIsFull() {
        KvCacheGroupPolicy full = KvCacheGroupPolicy.full();

        assertEquals(0, full.persistentStart(8));
        assertEquals(8, full.persistentTokenCount(8));
        assertEquals(0, full.attentionStart(8, 1));
        assertEquals(9, full.attentionEndExclusive(8, 1));
    }

    @Test
    void slidingGroupRetainsWindowMinusOneAndCurrentRowsAreVisible() {
        KvCacheGroupPolicy sliding = KvCacheGroupPolicy.sliding("sliding", 4);

        assertEquals(1, sliding.persistentStart(4));
        assertEquals(3, sliding.persistentTokenCount(4));
        assertEquals(1, sliding.attentionStart(4, 1));
        assertEquals(5, sliding.attentionEndExclusive(4, 1));
    }

    @Test
    void mixedLayerRoutingAndGroupSpecificIdentityDoNotAlias() {
        KvCacheGroupPolicy full = KvCacheGroupPolicy.full("full");
        KvCacheGroupPolicy sliding = KvCacheGroupPolicy.sliding("sliding", 4);
        KvCacheManager manager = new KvCacheManager(3, 32, 8, DType.F32,
                new KvBufferCacheSettings().withBlockSize(2), new MetricRegistry(), false,
                List.of(full, sliding), new int[] {0, 1, 0});

        assertSame(full, manager.groupForLayer(0));
        assertSame(sliding, manager.groupForLayer(1));
        assertSame(full, manager.groupForLayer(2));

        KvBlockKey block = KvBlockKey.local("model", "salt", 0, 0L, 1L, 2, 2,
                3, 8, DType.F32, DType.F32, KvBlockLayout.DENSE, 0);
        assertNotEquals(manager.groupBlockKey("full", block), manager.groupBlockKey("sliding", block));
    }

    @Test
    void groupSessionsHaveIndependentLengthAndPhysicalBlockOwnership() {
        KvCacheManager manager = new KvCacheManager(1, 32, 8, DType.F32,
                new KvBufferCacheSettings().withBlockSize(2), new MetricRegistry(), false,
                List.of(KvCacheGroupPolicy.full("full"), KvCacheGroupPolicy.sliding("sliding", 4)),
                new int[] {1});
        try (KvCacheSession full = manager.openSession("full");
             KvCacheSession sliding = manager.openSession("sliding")) {
            full.advanceLength(4);
            sliding.advanceLength(4);
            assertEquals(4, full.length());
            assertEquals(4, sliding.length());
            assertEquals(4, full.persistentTokenCount());
            assertEquals(3, sliding.persistentTokenCount());
            assertNotEquals(full.sessionId(), sliding.sessionId());
            assertEquals(List.of(), full.committedBlockIndexes());
            assertEquals(List.of(), sliding.committedBlockIndexes());
        }
    }

    @Test
    void readViewReportsPersistentAndCurrentAttentionRanges() {
        KvCacheManager manager = new KvCacheManager(1, 32, 8, DType.F32,
                new KvBufferCacheSettings().withBlockSize(2), new MetricRegistry(), false,
                List.of(KvCacheGroupPolicy.sliding("sliding", 4)), new int[] {0});
        try (KvCacheSession session = manager.openSessionForLayer(0)) {
            session.advanceLength(4);
            try (KvReadView view = session.readView(0, 4, AttentionPattern.CAUSAL, 1)) {
                assertEquals(1, view.persistentStartPosition());
                assertEquals(3, view.persistentTokenCount());
                assertEquals(1, view.attentionVisibleStart());
                assertEquals(5, view.attentionVisibleEndExclusive());
            }
        }
    }

    @Test
    void coordinatorRoutesRowsToLocalGroupStorageAndEvictsSlidingBlocks() {
        MetricRegistry metrics = new MetricRegistry();
        KvCacheManager manager = new KvCacheManager(3, 32, 8, DType.F32,
                new KvBufferCacheSettings().withBlockSize(2), metrics, false,
                List.of(KvCacheGroupPolicy.full("full"), KvCacheGroupPolicy.sliding("sliding", 4)),
                new int[] {0, 1, 0});
        Lighter lighter = new Lighter(metrics);
        try (KvCacheSession session = manager.openSession();
             TensorRef key = lighter.allocate(DType.F32, TensorShape.of(1, 8));
             TensorRef value = lighter.allocate(DType.F32, TensorShape.of(1, 8));
             KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
            for (int position = 0; position < 4; position++) {
                writer.write(0, position, key, value);
                writer.write(1, position, key, value);
                writer.write(2, position, key, value);
            }
            writer.advanceLength(4);

            KvCacheSession full = session.groupSession("full");
            KvCacheSession sliding = session.groupSession("sliding");
            KvBlock fullBlockZero = full.committedBlocks().get(0);
            KvBlock slidingBlockZero = sliding.committedBlocks().get(0);
            for (int position = 4; position < 6; position++) {
                writer.write(0, position, key, value);
                writer.write(1, position, key, value);
                writer.write(2, position, key, value);
            }
            writer.advanceLength(6);

            assertEquals(List.of(0, 1, 2), full.committedBlockIndexes());
            assertEquals(List.of(1, 2), sliding.committedBlockIndexes());
            assertFalse(fullBlockZero.isClosed());
            assertTrue(slidingBlockZero.isClosed());
            assertEquals(2, full.committedBlocks().get(0).layerCount());
            assertEquals(1, sliding.committedBlocks().get(0).layerCount());

            try (KvReadView fullView = session.readView(0, 6, AttentionPattern.CAUSAL);
                 KvReadView slidingView = session.readView(1, 6, AttentionPattern.CAUSAL)) {
                try (TensorRef ignored = fullView.keyRowCopyRef(0)) {
                    // The full group retains the row whose block was evicted from sliding storage.
                }
                assertThrows(IllegalArgumentException.class, () -> slidingView.keyRowCopyRef(0));
                try (TensorRef ignored = slidingView.keyRowCopyRef(3)) {
                    // The first persistent sliding row remains readable.
                }
                assertEquals(3, fullView.keyPageRefs().length);
                assertEquals(2, slidingView.keyPageRefs().length);
            }
        }
    }

    @Test
    void groupedSnapshotRestoresAbsoluteSuffixWithoutRestoringExpiredSlidingRows() {
        MetricRegistry metrics = new MetricRegistry();
        KvBufferCacheSettings settings = new KvBufferCacheSettings()
                .withBlockSize(2)
                .withMaxPrefixTokensPerPrompt(32)
                .withPrefixCacheMode(KvBufferCacheSettings.PrefixCacheMode.SNAPSHOT);
        KvCacheManager manager = groupedManager(settings, metrics);
        try (KvCacheSession source = manager.openSession();
             KvPrefixSnapshotCache snapshots = new KvPrefixSnapshotCache(3, 32, 8, 2,
                     DType.F32, DType.F32, new Lighter(metrics), metrics, settings)) {
            writeRows(source, metrics, 6);
            snapshots.storePrefix(new int[] {10, 11, 12, 13, 14, 15}, source, Optional.empty());

            try (KvCacheSession destination = manager.openSession()) {
                KvPrefixSnapshotCache.PrefixHit hit = snapshots.lookupPrefix(
                        new int[] {10, 11, 12, 13, 14, 15}, Optional.empty(), destination);
                assertEquals(6, hit.length());
                try (TensorRef ignored = destination.readView(0, 6, AttentionPattern.CAUSAL).keyRowCopyRef(0)) {
                    // Full attention restores the oldest prefix row.
                }
                KvReadView sliding = destination.readView(1, 6, AttentionPattern.CAUSAL);
                assertThrows(IllegalArgumentException.class, () -> sliding.keyRowCopyRef(0));
                try (TensorRef ignored = sliding.keyRowCopyRef(3)) {
                    // The suffix is restored at its original absolute position.
                }
                sliding.close();
            }
        }
    }

    @Test
    void groupScopedSharedBlocksDoNotAliasAndLeasesSurviveSessionClose() {
        MetricRegistry metrics = new MetricRegistry();
        KvBufferCacheSettings settings = new KvBufferCacheSettings().withBlockSize(2);
        KvCacheManager manager = groupedManager(settings, metrics);
        KvBlockManager shared = new KvBlockManager(metrics);
        KvBlockLease fullLease = null;
        KvBlockLease slidingLease = null;
        try (KvCacheSession source = manager.openSession()) {
            writeRows(source, metrics, 2);
            KvCacheSession full = source.groupSession("full");
            KvCacheSession sliding = source.groupSession("sliding");
            KvBlock fullBlock = full.detachCommittedBlock(0);
            KvBlock slidingBlock = sliding.detachCommittedBlock(0);
            KvBlockKey fullKey = KvBlockKey.local("model", "salt", 0, 0L, 99L, 2, 2,
                    2, 8, DType.F32, DType.F32, KvBlockLayout.DENSE, 0).withGroupId("full");
            KvBlockKey slidingKey = KvBlockKey.local("model", "salt", 0, 0L, 99L, 2, 2,
                    1, 8, DType.F32, DType.F32, KvBlockLayout.DENSE, 0).withGroupId("sliding");
            fullLease = shared.admitAndRetain(fullKey, fullBlock, "full-source");
            slidingLease = shared.admitAndRetain(slidingKey, slidingBlock, "sliding-source");
            assertNotEquals(fullLease.key(), slidingLease.key());
            assertEquals(2, shared.residentBlockCount());
        } finally {
            if (fullLease != null) fullLease.close();
            if (slidingLease != null) slidingLease.close();
            shared.close();
        }
    }

    private static KvCacheManager groupedManager(KvBufferCacheSettings settings, MetricRegistry metrics) {
        return new KvCacheManager(3, 32, 8, DType.F32, settings, metrics, false,
                List.of(KvCacheGroupPolicy.full("full"), KvCacheGroupPolicy.sliding("sliding", 4)),
                new int[] {0, 1, 0});
    }

    private static void writeRows(KvCacheSession session, MetricRegistry metrics, int length) {
        Lighter lighter = new Lighter(metrics);
        try (TensorRef key = lighter.allocate(DType.F32, TensorShape.of(1, 8));
             TensorRef value = lighter.allocate(DType.F32, TensorShape.of(1, 8));
             KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
            for (int position = 0; position < length; position++) {
                writer.write(0, position, key, value);
                writer.write(1, position, key, value);
                writer.write(2, position, key, value);
            }
            writer.advanceLength(length);
        }
    }
}
