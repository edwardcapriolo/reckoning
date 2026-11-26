package io.teknek.reckoning.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.model.InferenceProfiler;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/** Temporary snapshot prefix cache for KVCache2 sessions. */
public final class KvPrefixSnapshotCache implements AutoCloseable {
    public record CacheKey(Optional<String> salt, List<Integer> prefixTokens) {
        public CacheKey {
            salt = salt == null ? Optional.empty() : salt;
            prefixTokens = List.copyOf(prefixTokens);
        }
    }

    public record PrefixHit(int length) {
    }

    private interface StoredPrefixEntry extends AutoCloseable {
        int length();

        void restoreTo(KvCacheSession destination);

        @Override
        void close();
    }

    private final class DenseStoredPrefixEntry implements StoredPrefixEntry {
        private final int startPosition;
        private final int length;
        private final int snapshotLayers;
        private final TensorRef[] keysByLayer;
        private final TensorRef[] valuesByLayer;
        private final AtomicBoolean closed = new AtomicBoolean(false);

        private DenseStoredPrefixEntry(int startPosition, int length, int snapshotLayers, TensorRef[] keysByLayer,
                TensorRef[] valuesByLayer) {
            this.startPosition = startPosition;
            this.length = length;
            this.snapshotLayers = snapshotLayers;
            this.keysByLayer = keysByLayer;
            this.valuesByLayer = valuesByLayer;
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public void restoreTo(KvCacheSession destination) {
            if (closed.get()) {
                throw new IllegalStateException("prefix snapshot is closed");
            }
            long start = System.nanoTime();
            try {
                for (int row = 0; row < length; row++) {
                    int position = startPosition + row;
                    for (int layer = 0; layer < snapshotLayers; layer++) {
                        try (TensorRef key = keysByLayer[layer].slice(row);
                             TensorRef value = valuesByLayer[layer].slice(row)) {
                            destination.writeAbsoluteForRestore(layer, position, key, value);
                        }
                    }
                    if (destination.length() < position + 1) destination.advanceLength(position + 1);
                }
                destination.advanceLength(Math.max(destination.length(), startPosition + length));
            } finally {
                InferenceProfiler.timer(metricRegistry, "kvcache.v2.prefix.restore")
                        .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
            }
        }

        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) {
                for (TensorRef key : keysByLayer) {
                    key.close();
                }
                for (TensorRef value : valuesByLayer) {
                    value.close();
                }
            }
        }
    }

    private final class GroupedStoredPrefixEntry implements StoredPrefixEntry {
        private final int length;
        private final Map<String, DenseStoredPrefixEntry> entries;
        private final AtomicBoolean closed = new AtomicBoolean(false);

        private GroupedStoredPrefixEntry(int length, Map<String, DenseStoredPrefixEntry> entries) {
            this.length = length;
            this.entries = Map.copyOf(entries);
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public void restoreTo(KvCacheSession destination) {
            if (closed.get()) throw new IllegalStateException("grouped prefix snapshot is closed");
            Preconditions.checkState(destination.isGrouped(), "grouped prefix requires a grouped destination");
            destination.prepareGroupedRestore(length);
            for (Map.Entry<String, DenseStoredPrefixEntry> entry : entries.entrySet()) {
                entry.getValue().restoreTo(destination.groupSession(entry.getKey()));
            }
        }

        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) entries.values().forEach(DenseStoredPrefixEntry::close);
        }
    }

    private final int layers;
    private final int contextLength;
    private final int kvLength;
    private final int blockSize;
    private final DType keyDType;
    private final DType valueDType;
    private final Lighter lighter;
    private final MetricRegistry metricRegistry;
    private final KvBufferCacheSettings settings;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Map<CacheKey, StoredPrefixEntry> prefixCache;
    private volatile int[] layerKvLengths;

    public KvPrefixSnapshotCache(int layers, int contextLength, int kvLength, int blockSize, DType keyDType,
            DType valueDType,
            Lighter lighter, MetricRegistry metricRegistry, KvBufferCacheSettings settings) {
        Preconditions.checkArgument(layers > 0, "layers must be > 0");
        Preconditions.checkArgument(contextLength > 0, "contextLength must be > 0");
        Preconditions.checkArgument(kvLength > 0, "kvLength must be > 0");
        Preconditions.checkArgument(blockSize > 0, "blockSize must be > 0");
        this.layers = layers;
        this.contextLength = contextLength;
        this.kvLength = kvLength;
        this.blockSize = blockSize;
        this.keyDType = keyDType;
        this.valueDType = valueDType;
        this.lighter = lighter;
        this.metricRegistry = metricRegistry;
        this.settings = settings;
        this.prefixCache = Collections.synchronizedMap(new LinkedHashMap<CacheKey, StoredPrefixEntry>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<CacheKey, StoredPrefixEntry> eldest) {
                boolean evict = size() > KvPrefixSnapshotCache.this.settings.getMaxEntries();
                if (evict && eldest != null && eldest.getValue() != null) {
                    InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.evict").inc();
                    eldest.getValue().close();
                }
                return evict;
            }
        });
    }

    public void setLayerKvLengths(int[] lengths) {
        Preconditions.checkArgument(lengths != null && lengths.length == layers,
                "layer KV width count must equal layer count");
        int[] copy = lengths.clone();
        for (int width : copy) Preconditions.checkArgument(width > 0, "layer KV widths must be positive");
        layerKvLengths = copy;
    }

    public PrefixHit lookupPrefix(int[] tokens, Optional<String> salt, KvCacheSession destination) {
        requireOpen();
        if (settings.getPrefixCacheMode() != KvBufferCacheSettings.PrefixCacheMode.SNAPSHOT) {
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.lookup.skip.mode").inc();
            return null;
        }
        long start = System.nanoTime();
        try {
            synchronized (prefixCache) {
                StoredPrefixEntry best = null;
                int limit = settings.getMaxPrefixTokensPerPrompt();
                for (int prefixLen : checkpointLengths(Math.min(tokens.length, limit))) {
                    StoredPrefixEntry entry = prefixCache.get(new CacheKey(salt, prefixTokens(tokens, prefixLen)));
                    if (entry != null) {
                        best = entry;
                    }
                }
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.lookup").inc();
                if (best != null && best.length() >= blockSize && best.length() % blockSize == 0) {
                    best.restoreTo(destination);
                    InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.hits").inc();
                    return new PrefixHit(best.length());
                }
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.misses").inc();
                return null;
            }
        } finally {
            InferenceProfiler.timer(metricRegistry, "kvcache.v2.prefix.lookup.time")
                    .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        }
    }

    public void storePrefix(int[] tokens, KvCacheSession source, Optional<String> salt) {
        requireOpen();
        if (settings.getPrefixCacheMode() != KvBufferCacheSettings.PrefixCacheMode.SNAPSHOT) {
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.store.skip.mode").inc();
            return;
        }
        if (settings.getMaxEntries() < 1) {
            return;
        }
        long start = System.nanoTime();
        try {
            int limit = settings.getMaxPrefixTokensPerPrompt();
            for (int prefixLen : checkpointLengths(Math.min(tokens.length, limit))) {
                if (prefixLen > source.length()) {
                    continue;
                }
                CacheKey key = new CacheKey(salt, prefixTokens(tokens, prefixLen));
                synchronized (prefixCache) {
                    if (prefixCache.containsKey(key)) {
                        continue;
                    }
                }
                StoredPrefixEntry entry = snapshot(source, prefixLen);
                if (entry == null) {
                    continue;
                }
                boolean inserted = false;
                synchronized (prefixCache) {
                    if (!prefixCache.containsKey(key)) {
                        prefixCache.put(key, entry);
                        inserted = true;
                    }
                }
                if (!inserted) {
                    entry.close();
                }
            }
        } finally {
            InferenceProfiler.timer(metricRegistry, "kvcache.v2.prefix.store")
                    .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        }
    }

    List<Integer> checkpointLengths(int tokenLength) {
        if (tokenLength < blockSize) {
            return List.of();
        }
        int largest = (tokenLength / blockSize) * blockSize;
        if (largest < blockSize) {
            return List.of();
        }
        if (settings.getPrefixCheckpointPolicy() == KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS) {
            ArrayList<Integer> fixed = new ArrayList<>();
            for (int prefixLen = blockSize; prefixLen <= largest; prefixLen += blockSize) {
                fixed.add(prefixLen);
            }
            return fixed;
        }
        if (settings.getPrefixCheckpointPolicy() == KvBufferCacheSettings.PrefixCheckpointPolicy.START_AND_END) {
            int max = settings.getMaxPrefixCheckpointsPerPrompt();
            int startCount = (max + 1) / 2;
            int endCount = max - startCount;
            LinkedHashSet<Integer> selected = new LinkedHashSet<>();
            for (int prefixLen = blockSize; prefixLen <= largest && selected.size() < startCount; prefixLen += blockSize) {
                selected.add(prefixLen);
            }
            for (int prefixLen = largest - ((endCount - 1) * blockSize); prefixLen <= largest; prefixLen += blockSize) {
                if (prefixLen >= blockSize) {
                    selected.add(prefixLen);
                }
            }
            return selected.stream().sorted().toList();
        }
        int max = settings.getMaxPrefixCheckpointsPerPrompt();
        LinkedHashSet<Integer> selected = new LinkedHashSet<>();
        for (Integer anchor : settings.getPrefixCheckpointAnchors()) {
            int aligned = (anchor / blockSize) * blockSize;
            if (aligned >= blockSize && aligned <= largest) {
                selected.add(aligned);
            }
            if (selected.size() >= Math.max(0, max - 1)) {
                break;
            }
        }
        selected.add(largest);
        ArrayList<Integer> result = new ArrayList<>(selected);
        result.sort(Integer::compareTo);
        if (result.size() > max) {
            ArrayList<Integer> trimmed = new ArrayList<>(result.subList(0, max - 1));
            trimmed.add(largest);
            return trimmed.stream().distinct().sorted().toList();
        }
        return result;
    }

    private StoredPrefixEntry snapshot(KvCacheSession source, int prefixLen) {
        if (source.isGrouped()) return groupedSnapshot(source, prefixLen);
        return denseSnapshot(source, prefixLen, 0, layers);
    }

    private StoredPrefixEntry groupedSnapshot(KvCacheSession source, int prefixLen) {
        Map<String, DenseStoredPrefixEntry> entries = new LinkedHashMap<>();
        try {
            for (String groupId : source.groupIds()) {
                KvCacheSession group = source.groupSession(groupId);
                int start = group.groupPolicy().persistentStart(prefixLen);
                if (start < group.persistentStartPosition()) {
                    return null;
                }
                int firstBlock = start / blockSize;
                int alignedStart = firstBlock * blockSize;
                entries.put(groupId, denseSnapshot(group, prefixLen, alignedStart,
                        group.layerCount()));
            }
            return new GroupedStoredPrefixEntry(prefixLen, entries);
        } catch (RuntimeException | Error e) {
            entries.values().forEach(DenseStoredPrefixEntry::close);
            throw e;
        }
    }

    private DenseStoredPrefixEntry denseSnapshot(KvCacheSession source, int prefixLen, int startPosition,
            int snapshotLayers) {
        long start = System.nanoTime();
        int rowCount = prefixLen - startPosition;
        TensorRef[] keys = new TensorRef[snapshotLayers];
        TensorRef[] values = new TensorRef[snapshotLayers];
        try {
            for (int layer = 0; layer < snapshotLayers; layer++) {
                int width = source.layerKvLength(layer);
                keys[layer] = lighter.allocateDirty(keyDType, io.teknek.reckoning.tensor.TensorShape.of(rowCount, width));
                values[layer] = lighter.allocateDirty(valueDType, io.teknek.reckoning.tensor.TensorShape.of(rowCount, width));
                source.copyKeyRows(layer, startPosition, rowCount, keys[layer], 0);
                source.copyValueRows(layer, startPosition, rowCount, values[layer], 0);
            }
            long bytes = 0;
            for (int layer = 0; layer < snapshotLayers; layer++) {
                int width = source.layerKvLength(layer);
                bytes += (long) rowCount * width * (keyDType.size() + valueDType.size());
            }
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.snapshot.bytes").inc(bytes);
            return new DenseStoredPrefixEntry(startPosition, rowCount, snapshotLayers, keys, values);
        } catch (RuntimeException | Error e) {
            for (TensorRef key : keys) {
                if (key != null) {
                    key.close();
                }
            }
            for (TensorRef value : values) {
                if (value != null) {
                    value.close();
                }
            }
            throw e;
        } finally {
            InferenceProfiler.timer(metricRegistry, "kvcache.v2.prefix.snapshot")
                    .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        }
    }

    private int layerKvLength(int layer) {
        return layerKvLengths == null ? kvLength : layerKvLengths[layer];
    }

    private static List<Integer> prefixTokens(int[] tokens, int prefixLen) {
        ArrayList<Integer> prefix = new ArrayList<>(prefixLen);
        for (int i = 0; i < prefixLen; i++) {
            prefix.add(tokens[i]);
        }
        return prefix;
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KVCache2 prefix snapshot cache is closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            synchronized (prefixCache) {
                prefixCache.values().forEach(StoredPrefixEntry::close);
                prefixCache.clear();
            }
        }
    }
}
