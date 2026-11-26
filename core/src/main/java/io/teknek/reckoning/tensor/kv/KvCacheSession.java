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
import java.util.HashMap;
import java.util.List;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.TreeMap;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/** Request-local KV cache v2 session with immutable committed blocks and mutable active blocks. */
public final class KvCacheSession implements AutoCloseable {
    private static final AtomicLong SESSION_SEQUENCE = new AtomicLong();

    private final String sessionId;
    private final int layers;
    private final int contextLength;
    private final int kvLength;
    private final int blockSize;
    private final DType dtype;
    private final MetricRegistry metricRegistry;
    private final boolean trackReadViews;
    private final KvBufferCacheSettings settings;
    private final Lighter lighter;
    private final int[] layerKvLengths;
    private final KvCacheGroupPolicy groupPolicy;
    private final Map<String, KvCacheSession> groupSessions;
    private final String[] groupIdsByIndex;
    private final int[] layerGroupIndexes;
    private final int[] layerLocalIndexes;
    private final NavigableMap<Integer, KvBlock> committedBlocks = new TreeMap<>();
    private final NavigableMap<Integer, KvBlockLease> committedBlockLeases = new TreeMap<>();
    private final NavigableMap<Integer, MutableKvBlock> mutableBlocks = new TreeMap<>();
    private final HashMap<Class<?>, AutoCloseable> attachments = new HashMap<>();
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private int length;

    KvCacheSession(int layers, int contextLength, int kvLength, int blockSize, DType dtype,
            MetricRegistry metricRegistry, boolean trackReadViews,
            KvBufferCacheSettings settings, Lighter lighter,
            int[] layerKvLengths) {
        this(layers, contextLength, kvLength, blockSize, dtype, metricRegistry, trackReadViews,
                settings, lighter, layerKvLengths, KvCacheGroupPolicy.full());
    }

    KvCacheSession(int layers, int contextLength, int kvLength, int blockSize, DType dtype,
            MetricRegistry metricRegistry, boolean trackReadViews,
            KvBufferCacheSettings settings, Lighter lighter, int[] layerKvLengths,
            KvCacheGroupPolicy groupPolicy) {
        this.sessionId = "kv2-session-" + SESSION_SEQUENCE.incrementAndGet();
        this.layers = layers;
        this.contextLength = contextLength;
        this.kvLength = kvLength;
        this.blockSize = blockSize;
        this.dtype = dtype;
        this.metricRegistry = metricRegistry;
        this.trackReadViews = trackReadViews;
        this.settings = settings;
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.layerKvLengths = layerKvLengths == null ? null : layerKvLengths.clone();
        this.groupPolicy = Objects.requireNonNull(groupPolicy, "groupPolicy");
        this.groupSessions = Map.of();
        this.groupIdsByIndex = null;
        this.layerGroupIndexes = null;
        this.layerLocalIndexes = null;
    }

    KvCacheSession(int layers, int contextLength, int kvLength, int blockSize, DType dtype,
            MetricRegistry metricRegistry, boolean trackReadViews, KvBufferCacheSettings settings,
            Lighter lighter, int[] layerKvLengths, List<KvCacheGroupPolicy> policies,
            int[] layerGroupIndexes) {
        this.sessionId = "kv2-session-" + SESSION_SEQUENCE.incrementAndGet();
        this.layers = layers;
        this.contextLength = contextLength;
        this.kvLength = kvLength;
        this.blockSize = blockSize;
        this.dtype = dtype;
        this.metricRegistry = metricRegistry;
        this.trackReadViews = trackReadViews;
        this.settings = settings;
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.layerKvLengths = layerKvLengths == null ? null : layerKvLengths.clone();
        this.groupPolicy = KvCacheGroupPolicy.full("coordinator");
        this.layerGroupIndexes = layerGroupIndexes.clone();
        this.groupIdsByIndex = new String[policies.size()];
        this.layerLocalIndexes = new int[layers];
        LinkedHashMap<String, List<Integer>> layersByGroup = new LinkedHashMap<>();
        for (KvCacheGroupPolicy policy : policies) {
            layersByGroup.put(policy.groupId(), new ArrayList<>());
        }
        for (int layer = 0; layer < layers; layer++) {
            int groupIndex = layerGroupIndexes[layer];
            List<Integer> localLayers = layersByGroup.get(policies.get(groupIndex).groupId());
            layerLocalIndexes[layer] = localLayers.size();
            localLayers.add(layer);
        }
        LinkedHashMap<String, KvCacheSession> children = new LinkedHashMap<>();
        for (int groupIndex = 0; groupIndex < policies.size(); groupIndex++) {
            KvCacheGroupPolicy policy = policies.get(groupIndex);
            groupIdsByIndex[groupIndex] = policy.groupId();
            List<Integer> globalLayers = layersByGroup.get(policy.groupId());
            int[] localWidths = null;
            if (layerKvLengths != null) {
                localWidths = new int[globalLayers.size()];
                for (int local = 0; local < globalLayers.size(); local++) {
                    localWidths[local] = layerKvLengths[globalLayers.get(local)];
                }
            }
            int localKvLength = localWidths == null ? kvLength : 0;
            if (localWidths != null) {
                for (int width : localWidths) localKvLength = Math.max(localKvLength, width);
            }
            children.put(policy.groupId(), new KvCacheSession(globalLayers.size(), contextLength,
                    localKvLength, blockSize, dtype, metricRegistry, trackReadViews, settings,
                    lighter, localWidths, policy));
        }
        this.groupSessions = Map.copyOf(children);
    }

    KvCacheSession(int layers, int contextLength, int kvLength, int blockSize, DType dtype,
            MetricRegistry metricRegistry, boolean trackReadViews, KvBufferCacheSettings settings,
            Lighter lighter) {
        this(layers, contextLength, kvLength, blockSize, dtype, metricRegistry, trackReadViews,
                settings, lighter, null);
    }

    public int length() {
        return length;
    }

    public String sessionId() {
        return sessionId;
    }

    public KvCacheGroupPolicy groupPolicy() {
        return groupPolicy;
    }

    public boolean isGrouped() {
        return !groupSessions.isEmpty();
    }

    public KvCacheSession groupSession(String groupId) {
        KvCacheSession session = groupSessions.get(Objects.requireNonNull(groupId, "groupId"));
        return Objects.requireNonNull(session, "unknown group: " + groupId);
    }

    public KvCacheGroupPolicy groupPolicyForLayer(int layer) {
        if (!isGrouped()) {
            validateLayer(layer);
            return groupPolicy;
        }
        validateLayer(layer);
        return groupSessionAt(layer).groupPolicy;
    }

    public List<String> groupIds() {
        if (!isGrouped()) return List.of(groupPolicy.groupId());
        return List.copyOf(java.util.Arrays.asList(groupIdsByIndex));
    }

    public int layerCount() {
        return layers;
    }

    public int kvLength() {
        return kvLength;
    }

    /** Sets absolute length before attaching grouped retained blocks. */
    public void prepareGroupedRestore(int restoredLength) {
        requireOpen();
        Preconditions.checkState(isGrouped(), "only grouped sessions require grouped restore");
        Preconditions.checkArgument(restoredLength >= 0 && restoredLength <= contextLength,
                "restored length out of bounds");
        for (KvCacheSession group : groupSessions.values()) group.setLengthForRestore(restoredLength);
        length = restoredLength;
    }

    private void setLengthForRestore(int restoredLength) {
        requireOpen();
        Preconditions.checkArgument(restoredLength >= length && restoredLength <= contextLength,
                "restored length out of bounds");
        length = restoredLength;
    }

    public int persistentStartPosition() {
        Preconditions.checkState(!isGrouped(), "grouped sessions require a layer or group id");
        return groupPolicy.persistentStart(length);
    }

    public int persistentTokenCount() {
        Preconditions.checkState(!isGrouped(), "grouped sessions require a layer or group id");
        return groupPolicy.persistentTokenCount(length);
    }

    public List<Integer> committedBlockIndexes() {
        Preconditions.checkState(!isGrouped(), "grouped sessions require a group session");
        return List.copyOf(committedBlocks.keySet());
    }

    public int blockSize() {
        return blockSize;
    }

    int layerKvLength(int layer) {
        if (isGrouped()) return groupSessionAt(layer).layerKvLength(layerLocalIndexes[layer]);
        return layerKvLengths == null ? kvLength : layerKvLengths[layer];
    }

    public boolean isHeterogeneous() {
        if (isGrouped()) return groupSessionAt(0).isHeterogeneous();
        return layerKvLengths != null;
    }

    DType keyDType() {
        return settings.getKvKeyDType();
    }

    DType valueDType() {
        return settings.getKvValueDType();
    }

    public List<KvBlock> committedBlocks() {
        Preconditions.checkState(!isGrouped(), "grouped sessions require a group session");
        return List.copyOf(committedBlocks.values());
    }

    /** Returns the request-local attachment of {@code type}, or {@code null} when absent. */
    public <T extends AutoCloseable> T attachment(Class<T> type) {
        requireOpen();
        return type.cast(attachments.get(Objects.requireNonNull(type, "type")));
    }

    /** Installs a request-local attachment. Replacing one closes the previous value. */
    public <T extends AutoCloseable> T attach(Class<T> type, T attachment) {
        requireOpen();
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(attachment, "attachment");
        AutoCloseable previous = attachments.put(type, attachment);
        if (previous != null) {
            closeAttachment(previous);
        }
        return attachment;
    }

    public <T extends AutoCloseable> T getOrAttach(Class<T> type, java.util.function.Supplier<T> factory) {
        requireOpen();
        T current = attachment(type);
        return current != null ? current : attach(type, factory.get());
    }

    List<KvBlockLease> committedBlockLeases() {
        return List.copyOf(committedBlockLeases.values());
    }

    public KvWriteCursor writer(CacheExecutionMode mode) {
        requireOpen();
        return new KvWriteCursor(this, Objects.requireNonNull(mode, "mode"));
    }

    public KvReadView readView(int layer, int visibleTokens, AttentionPattern pattern) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(visibleTokens >= 0 && visibleTokens <= length,
                "visibleTokens must be within session length");
        return new KvReadView(this, layer, visibleTokens, Objects.requireNonNull(pattern, "pattern"), 0);
    }

    public KvReadView readView(int layer, int visibleTokens, AttentionPattern pattern, int currentRows) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(visibleTokens >= 0 && visibleTokens <= length,
                "visibleTokens must be within session length");
        Preconditions.checkArgument(currentRows >= 0 && visibleTokens + currentRows <= contextLength,
                "current rows exceed context length");
        return new KvReadView(this, layer, visibleTokens, Objects.requireNonNull(pattern, "pattern"), currentRows);
    }

    TensorRef keyRowCopyRef(int layer, int position) {
        if (isGrouped()) return groupSessionAt(layer).keyRowCopyRef(layerLocalIndexes[layer], position);
        return rowCopyRef(layer, position, true);
    }

    TensorRef valueRowCopyRef(int layer, int position) {
        if (isGrouped()) return groupSessionAt(layer).valueRowCopyRef(layerLocalIndexes[layer], position);
        return rowCopyRef(layer, position, false);
    }

    void write(CacheExecutionMode mode, int layer, int position, TensorRef key, TensorRef value) {
        requireOpen();
        if (isGrouped()) {
            groupSessionAt(layer).write(mode, layerLocalIndexes[layer], position, key, value);
            return;
        }
        writeLocal(mode, layer, position, key, value, false);
    }

    void writeAbsoluteForRestore(int layer, int position, TensorRef key, TensorRef value) {
        requireOpen();
        Preconditions.checkState(!isGrouped(), "restore rows must target a group session");
        writeLocal(CacheExecutionMode.PREFILL_UPDATE_CACHE, layer, position, key, value, true);
    }

    private void writeLocal(CacheExecutionMode mode, int layer, int position, TensorRef key, TensorRef value,
            boolean restore) {
        Preconditions.checkArgument(mode != CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE
                        && mode != CacheExecutionMode.READ_PREFIX_NO_UPDATE,
                mode + " must not write KV rows");
        validateLayer(layer);
        if (restore) {
            Preconditions.checkArgument(position >= 0 && position < contextLength, "position out of bounds");
        } else {
            validatePosition(position);
        }
        int blockIndex = position / blockSize;
        Preconditions.checkArgument(!committedBlocks.containsKey(blockIndex), "cannot overwrite committed KV block");
        TensorRef keyToWrite = key;
        TensorRef valueToWrite = value;
        TensorRef convertedKey = null;
        TensorRef convertedValue = null;
        try {
            if (key.dType() != settings.getKvKeyDType()) {
                convertedKey = lighter.reshape(key, settings.getKvKeyDType());
                keyToWrite = convertedKey;
            }
            if (value.dType() != settings.getKvValueDType()) {
                convertedValue = lighter.reshape(value, settings.getKvValueDType());
                valueToWrite = convertedValue;
            }
            mutableBlock(blockIndex).write(layer, position, keyToWrite, valueToWrite);
            metricRegistry.meter("kvcache.v2.row.write").mark();
        } finally {
            if (convertedKey != null) convertedKey.close();
            if (convertedValue != null) convertedValue.close();
        }
    }

    public void advanceLength(int newLength) {
        requireOpen();
        Preconditions.checkArgument(newLength >= length, "newLength must be >= current length");
        Preconditions.checkArgument(newLength <= contextLength, "newLength exceeds context length");
        if (isGrouped()) {
            for (KvCacheSession group : groupSessions.values()) {
                group.advanceLength(newLength);
            }
            length = newLength;
        } else {
            length = newLength;
            commitFullBlocksBefore(length);
            evictSlidingBlocks(length);
        }
    }

    public void crop(int newLength) {
        requireOpen();
        Preconditions.checkArgument(newLength >= 0 && newLength <= length, "newLength out of bounds");
        if (isGrouped()) {
            for (KvCacheSession group : groupSessions.values()) group.crop(newLength);
            length = newLength;
            return;
        }
        int lastBlockToKeep = newLength == 0 ? -1 : (newLength - 1) / blockSize;
        int rowsInLastBlock = newLength == 0 ? 0 : ((newLength - 1) % blockSize) + 1;

        ArrayList<Integer> committedToRemove = new ArrayList<>(committedBlocks.tailMap(lastBlockToKeep + 1, true).keySet());
        for (Integer blockIndex : committedToRemove) {
            closeCommittedBlock(committedBlocks.get(blockIndex), committedBlockLeases.get(blockIndex));
        }
        committedBlocks.tailMap(lastBlockToKeep + 1, true).clear();
        committedBlockLeases.tailMap(lastBlockToKeep + 1, true).clear();
        mutableBlocks.tailMap(lastBlockToKeep + 1, true).values().forEach(MutableKvBlock::close);
        mutableBlocks.tailMap(lastBlockToKeep + 1, true).clear();
        if (rowsInLastBlock > 0 && rowsInLastBlock < blockSize) {
            splitCommittedTailBlock(lastBlockToKeep, newLength);
        }
        length = newLength;
        evictSlidingBlocks(length);
        metricRegistry.meter("kvcache.v2.crop").mark();
    }

    private void splitCommittedTailBlock(int blockIndex, int newLength) {
        KvBlock committed = committedBlocks.remove(blockIndex);
        if (committed == null) {
            return;
        }
        KvBlockLease lease = committedBlockLeases.remove(blockIndex);
        MutableKvBlock mutable = new MutableKvBlock(blockIndex, blockSize, layers, kvLength, dtype,
                settings, metricRegistry, lighter, layerKvLengths);
        try {
            for (int position = committed.startPosition(); position < newLength; position++) {
                for (int layer = 0; layer < layers; layer++) {
                        try (TensorRef key = committed.keyRowCopy(layer, position, lighter);
                             TensorRef value = committed.valueRowCopy(layer, position, lighter)) {
                        mutable.write(layer, position, key, value);
                    }
                }
            }
            closeCommittedBlock(committed, lease);
            MutableKvBlock previous = mutableBlocks.put(blockIndex, mutable);
            if (previous != null) {
                previous.close();
            }
        } catch (RuntimeException | Error e) {
            mutable.close();
            closeCommittedBlock(committed, lease);
            throw e;
        }
    }

    /** Transfers an owned committed block out of this session for shared-block admission. */
    public KvBlock detachCommittedBlock(int blockIndex) {
        requireOpen();
        Preconditions.checkState(!isGrouped(), "grouped sessions require group-specific prefix ownership");
        KvBlockLease lease = committedBlockLeases.remove(blockIndex);
        Preconditions.checkState(lease == null, "cannot detach a leased KV block");
        KvBlock block = committedBlocks.remove(blockIndex);
        Preconditions.checkState(block != null, "no committed KV block at index " + blockIndex);
        return block;
    }

    public KvBlock detachCommittedBlock(String groupId, int blockIndex) {
        Preconditions.checkState(isGrouped(), "group-specific ownership requires a grouped session");
        return groupSession(groupId).detachCommittedBlock(blockIndex);
    }

    /** Attaches a retained shared immutable block lease to this session. */
    public void attachCommittedBlock(KvBlockLease lease) {
        requireOpen();
        Preconditions.checkState(!isGrouped(), "grouped sessions require group-specific prefix ownership");
        Objects.requireNonNull(lease, "lease");
        KvBlock block = lease.block();
        Preconditions.checkArgument(block.tokenCount() == blockSize,
                "shared KV block must be a full committed block");
        Preconditions.checkArgument(block.blockSize() == blockSize, "block size mismatch");
        Preconditions.checkArgument(block.endPositionExclusive() <= length || block.startPosition() == length,
                "shared KV blocks must attach contiguously or inside current length");
        Preconditions.checkArgument(!mutableBlocks.containsKey(block.blockIndex()),
                "cannot attach shared block over mutable block");
        Preconditions.checkArgument(!committedBlocks.containsKey(block.blockIndex()),
                "cannot attach shared block over committed block");
        committedBlocks.put(block.blockIndex(), block);
        committedBlockLeases.put(block.blockIndex(), lease);
        length = Math.max(length, block.endPositionExclusive());
        metricRegistry.meter("kvcache.v2.block.attach").mark();
    }

    public void attachCommittedBlock(String groupId, KvBlockLease lease) {
        Preconditions.checkState(isGrouped(), "group-specific ownership requires a grouped session");
        groupSession(groupId).attachCommittedBlock(lease);
    }

    List<Integer> attachedLeaseBlockIndexes() {
        return Collections.unmodifiableList(new ArrayList<>(committedBlockLeases.keySet()));
    }

    public boolean committedBlockIsLeased(int blockIndex) {
        Preconditions.checkState(!isGrouped(), "grouped sessions require group-specific prefix ownership");
        return committedBlockLeases.containsKey(blockIndex);
    }

    public boolean committedBlockIsLeased(String groupId, int blockIndex) {
        Preconditions.checkState(isGrouped(), "group-specific ownership requires a grouped session");
        return groupSession(groupId).committedBlockIsLeased(blockIndex);
    }

    TensorRef copyVisibleKeysRef(int layer, int visibleTokens) {
        if (isGrouped()) return groupSessionAt(layer).copyVisibleKeysRef(layerLocalIndexes[layer], visibleTokens);
        return copyVisible(layer, visibleTokens, true);
    }

    TensorRef copyVisibleValuesRef(int layer, int visibleTokens) {
        if (isGrouped()) return groupSessionAt(layer).copyVisibleValuesRef(layerLocalIndexes[layer], visibleTokens);
        return copyVisible(layer, visibleTokens, false);
    }

    void copyKeyRows(int layer, int positionStart, int rowCount, TensorRef destination, int destinationRowStart) {
        if (isGrouped()) {
            groupSessionAt(layer).copyKeyRows(layerLocalIndexes[layer], positionStart, rowCount, destination, destinationRowStart);
            return;
        }
        copyRows(layer, positionStart, rowCount, destination, destinationRowStart, true);
    }

    void copyValueRows(int layer, int positionStart, int rowCount, TensorRef destination, int destinationRowStart) {
        if (isGrouped()) {
            groupSessionAt(layer).copyValueRows(layerLocalIndexes[layer], positionStart, rowCount, destination, destinationRowStart);
            return;
        }
        copyRows(layer, positionStart, rowCount, destination, destinationRowStart, false);
    }

    TensorRef[] keyPageRefs(int layer, int visibleTokens) {
        if (isGrouped()) return groupSessionAt(layer).keyPageRefs(layerLocalIndexes[layer], visibleTokens);
        return pages(layer, visibleTokens, true);
    }

    TensorRef[] valuePageRefs(int layer, int visibleTokens) {
        if (isGrouped()) return groupSessionAt(layer).valuePageRefs(layerLocalIndexes[layer], visibleTokens);
        return pages(layer, visibleTokens, false);
    }

    private TensorRef copyVisible(int layer, int visibleTokens, boolean key) {
        TensorRef result = lighter.allocateDirty(dtype,
                io.teknek.reckoning.tensor.TensorShape.of(visibleTokens, layerKvLength(layer)));
        try {
            for (int position = 0; position < visibleTokens; position++) {
                try (TensorRef row = rowCopyRef(layer, position, key)) {
                    lighter.copyFrom(row, result, 0, result.shape().getOffset(position, 0), layerKvLength(layer));
                }
            }
            return result;
        } catch (RuntimeException | Error e) {
            result.close();
            throw e;
        }
    }

    private void copyRows(int layer, int positionStart, int rowCount, TensorRef destination, int destinationRowStart,
            boolean key) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(positionStart >= 0 && rowCount >= 0 && positionStart + rowCount <= length,
                "position range out of visible length");
        Preconditions.checkArgument(destination.dims() == 2 && destination.shape().last() == layerKvLength(layer),
                "destination must have layer KV width columns");
        Preconditions.checkArgument(destinationRowStart >= 0
                && destinationRowStart + rowCount <= destination.shape().first(), "destination row range out of bounds");
        int remaining = rowCount;
        int sourcePosition = positionStart;
        int destRow = destinationRowStart;
        while (remaining > 0) {
            int blockIndex = sourcePosition / blockSize;
            int blockRow = sourcePosition % blockSize;
            int rowsInBlock = Math.min(remaining, blockSize - blockRow);
            KvBlock committed = committedBlocks.get(blockIndex);
            if (committed != null) {
                rowsInBlock = Math.min(rowsInBlock, committed.endPositionExclusive() - sourcePosition);
                if (key) {
                    committed.copyKeyRows(layer, sourcePosition, rowsInBlock, destination, destRow);
                } else {
                    committed.copyValueRows(layer, sourcePosition, rowsInBlock, destination, destRow);
                }
            } else {
                MutableKvBlock mutable = mutableBlocks.get(blockIndex);
                if (mutable == null) {
                    throw new IllegalStateException("No KV block for position " + sourcePosition);
                }
                if (key) {
                    mutable.copyKeyRows(layer, sourcePosition, rowsInBlock, destination, destRow);
                } else {
                    mutable.copyValueRows(layer, sourcePosition, rowsInBlock, destination, destRow);
                }
            }
            remaining -= rowsInBlock;
            sourcePosition += rowsInBlock;
            destRow += rowsInBlock;
        }
    }

    private TensorRef[] pages(int layer, int visibleTokens, boolean key) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(visibleTokens >= 0 && visibleTokens <= length,
                "visibleTokens must be within session length");
        if (visibleTokens == 0) {
            return new TensorRef[0];
        }
        int firstBlock = groupPolicy.type() == KvCacheGroupType.SLIDING
                ? groupPolicy.persistentStart(length) / blockSize : 0;
        int pageCount = Math.max(0, (visibleTokens + blockSize - 1) / blockSize - firstBlock);
        // TensorRef-backed pages are caller-owned views and are closed after each attention pass.
        return buildPages(layer, firstBlock, pageCount, key);
    }

    private TensorRef[] buildPages(int layer, int firstBlock, int pageCount, boolean key) {
        TensorRef[] pages = new TensorRef[pageCount];
        try {
            for (int page = 0; page < pageCount; page++) {
                int blockIndex = firstBlock + page;
                KvBlock committed = committedBlocks.get(blockIndex);
                if (committed != null) {
                    pages[page] = key ? committed.keyPageView(layer) : committed.valuePageView(layer);
                } else {
                    MutableKvBlock mutable = mutableBlocks.get(blockIndex);
                    if (mutable == null) {
                        throw new IllegalStateException("No KV block for block index " + blockIndex);
                    }
                    pages[page] = key ? mutable.keyPageView(layer) : mutable.valuePageView(layer);
                }
            }
            return pages;
        } catch (RuntimeException | Error e) {
            for (TensorRef page : pages) {
                if (page != null) {
                    page.close();
                }
            }
            throw e;
        }
    }

    private TensorRef rowCopyRef(int layer, int position, boolean key) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(position >= 0 && position < length, "position out of visible length");
        validateReadablePosition(position);
        int blockIndex = position / blockSize;
        KvBlock committed = committedBlocks.get(blockIndex);
        if (committed != null) {
            return key ? committed.keyRowCopy(layer, position, lighter) : committed.valueRowCopy(layer, position, lighter);
        }
        MutableKvBlock mutable = mutableBlocks.get(blockIndex);
        if (mutable == null) {
            throw new IllegalStateException("No KV block for position " + position);
        }
        return key ? mutable.keyRowCopy(layer, position, lighter) : mutable.valueRowCopy(layer, position, lighter);
    }

    TensorRef keyRowViewRef(int layer, int position) {
        return rowViewRef(layer, position, true);
    }

    TensorRef valueRowViewRef(int layer, int position) {
        return rowViewRef(layer, position, false);
    }

    private TensorRef rowViewRef(int layer, int position, boolean key) {
        requireOpen();
        validateLayer(layer);
        if (isGrouped()) return groupSessionAt(layer).rowViewRef(layerLocalIndexes[layer], position, key);
        Preconditions.checkArgument(position >= 0 && position < length, "position out of visible length");
        validateReadablePosition(position);
        if (isHeterogeneous()) {
            return rowCopyRef(layer, position, key);
        }
        int blockIndex = position / blockSize;
        TensorRef view;
        KvBlock committed = committedBlocks.get(blockIndex);
        if (committed != null) {
            view = key ? committed.keyRowView(layer, position) : committed.valueRowView(layer, position);
        } else {
            MutableKvBlock mutable = mutableBlocks.get(blockIndex);
            if (mutable == null) {
                throw new IllegalStateException("No KV block for position " + position);
            }
            view = key ? mutable.keyRowView(layer, position) : mutable.valueRowView(layer, position);
        }
        return view;
    }

    private MutableKvBlock mutableBlock(int blockIndex) {
        return mutableBlocks.computeIfAbsent(blockIndex,
                index -> new MutableKvBlock(index, blockSize, layers, kvLength, dtype, settings,
                        metricRegistry, lighter, layerKvLengths));
    }

    private void commitFullBlocksBefore(int newLength) {
        int fullBlocks = newLength / blockSize;
        List<Integer> toCommit = new ArrayList<>(mutableBlocks.headMap(fullBlocks, false).keySet());
        for (Integer blockIndex : toCommit) {
            MutableKvBlock mutable = mutableBlocks.remove(blockIndex);
            committedBlocks.put(blockIndex, mutable.commit(blockSize));
            metricRegistry.meter("kvcache.v2.block.commit").mark();
        }
    }

    /** Removes only complete blocks that no longer contain a persistent sliding row. */
    private void evictSlidingBlocks(int newLength) {
        if (groupPolicy.type() != KvCacheGroupType.SLIDING) return;
        int firstPersistentPosition = groupPolicy.persistentStart(newLength);
        int firstRetainedBlock = firstPersistentPosition / blockSize;
        ArrayList<Integer> committedToRemove = new ArrayList<>(
                committedBlocks.headMap(firstRetainedBlock, false).keySet());
        for (Integer blockIndex : committedToRemove) {
            closeCommittedBlock(committedBlocks.remove(blockIndex), committedBlockLeases.remove(blockIndex));
            metricRegistry.meter("kvcache.v2.sliding.block.evict").mark();
        }
        ArrayList<Integer> mutableToRemove = new ArrayList<>(
                mutableBlocks.headMap(firstRetainedBlock, false).keySet());
        for (Integer blockIndex : mutableToRemove) {
            MutableKvBlock block = mutableBlocks.remove(blockIndex);
            if (block != null) block.close();
            metricRegistry.meter("kvcache.v2.sliding.block.evict").mark();
        }
    }

    private void closeCommittedBlock(KvBlock block, KvBlockLease lease) {
        if (lease != null) {
            lease.close();
        } else {
            block.close();
        }
    }

    private void validateLayer(int layer) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
    }

    private void validatePosition(int position) {
        Preconditions.checkArgument(position >= 0 && position < contextLength, "position out of bounds");
        Preconditions.checkArgument(position >= groupPolicy.persistentStart(length),
                "position is outside the persistent %s KV range", groupPolicy.groupId());
    }

    private void validateReadablePosition(int position) {
        Preconditions.checkArgument(position >= groupPolicy.persistentStart(length),
                "position is outside the persistent %s KV range", groupPolicy.groupId());
    }

    private KvCacheSession groupSessionAt(int layer) {
        validateLayer(layer);
        int groupIndex = layerGroupIndexes[layer];
        return groupSessions.get(groupIdsByIndex[groupIndex]);
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KV cache session is closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            groupSessions.values().forEach(KvCacheSession::close);
            for (var entry : committedBlocks.entrySet()) {
                closeCommittedBlock(entry.getValue(), committedBlockLeases.get(entry.getKey()));
            }
            mutableBlocks.values().forEach(MutableKvBlock::close);
            committedBlocks.clear();
            committedBlockLeases.clear();
            mutableBlocks.clear();
            attachments.values().forEach(this::closeAttachment);
            attachments.clear();
            metricRegistry.meter("kvcache.v2.session.close").mark();
        }
    }

    private void closeAttachment(AutoCloseable attachment) {
        try {
            attachment.close();
        } catch (Exception e) {
            throw new IllegalStateException("Unable to close KV session attachment", e);
        }
    }
}
