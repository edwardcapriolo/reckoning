package io.teknek.reckoning.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor2.Lighter;

import java.util.Objects;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;

/** Opens standalone KV cache v2 sessions. */
public final class KvCacheManager {
    private final int layers;
    private final int contextLength;
    private int kvLength;
    private final int blockSize;
    private final DType dtype;
    private final MetricRegistry metricRegistry;
    private final boolean trackReadViews;
    private final KvBufferCacheSettings settings;
    private final Lighter lighter;
    private volatile int[] layerKvLengths;
    private volatile Map<String, KvCacheGroupPolicy> groupPolicies;
    private volatile List<KvCacheGroupPolicy> orderedGroupPolicies;
    private volatile int[] layerGroupIndexes;

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, MetricRegistry metricRegistry,
            boolean trackReadViews) {
        this(layers, contextLength, kvLength, dtype, settings, metricRegistry, trackReadViews,
                new Lighter(metricRegistry));
    }

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, MetricRegistry metricRegistry,
            boolean trackReadViews, Lighter lighter) {
        Preconditions.checkArgument(layers > 0, "layers must be > 0");
        Preconditions.checkArgument(contextLength > 0, "contextLength must be > 0");
        Preconditions.checkArgument(kvLength > 0, "kvLength must be > 0");
        this.layers = layers;
        this.contextLength = contextLength;
        this.kvLength = kvLength;
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        this.settings = Objects.requireNonNull(settings, "settings");
        this.blockSize = this.settings.getBlockSize();
        // The TensorRef-backed path allocates through Lighter and intentionally has no V1 allocator.
        this.metricRegistry = Objects.requireNonNull(metricRegistry, "metricRegistry");
        this.trackReadViews = trackReadViews;
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.groupPolicies = Map.of("full", KvCacheGroupPolicy.full());
        this.orderedGroupPolicies = List.of(KvCacheGroupPolicy.full());
    }

    /** Creates a manager with explicit physical group policies and layer routing. */
    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, MetricRegistry metricRegistry, boolean trackReadViews,
            List<KvCacheGroupPolicy> policies, int[] layerGroupIndexes) {
        this(layers, contextLength, kvLength, dtype, settings, metricRegistry, trackReadViews,
                new Lighter(metricRegistry), policies, layerGroupIndexes);
    }

    public KvCacheManager(int layers, int contextLength, int kvLength, DType dtype,
            KvBufferCacheSettings settings, MetricRegistry metricRegistry, boolean trackReadViews,
            Lighter lighter, List<KvCacheGroupPolicy> policies, int[] layerGroupIndexes) {
        this(layers, contextLength, kvLength, dtype, settings, metricRegistry, trackReadViews, lighter);
        Preconditions.checkArgument(policies != null && !policies.isEmpty(), "at least one group is required");
        HashMap<String, KvCacheGroupPolicy> byId = new HashMap<>();
        policies.forEach(policy -> {
            Preconditions.checkArgument(byId.put(policy.groupId(), policy) == null,
                    "duplicate KV cache group: %s", policy.groupId());
        });
        Preconditions.checkArgument(layerGroupIndexes != null && layerGroupIndexes.length == layers,
                "layer group indexes must equal layer count");
        for (int index : layerGroupIndexes) {
            Preconditions.checkArgument(index >= 0 && index < policies.size(), "layer group index out of bounds");
        }
        this.groupPolicies = Map.copyOf(byId);
        this.orderedGroupPolicies = List.copyOf(policies);
        this.layerGroupIndexes = layerGroupIndexes.clone();
    }

    public void setLayerKvLengths(int[] lengths) {
        Preconditions.checkArgument(lengths != null && lengths.length == layers,
                "layer KV width count must equal layer count");
        int[] copy = lengths.clone();
        int maximum = 0;
        for (int length : copy) {
            Preconditions.checkArgument(length > 0, "layer KV widths must be positive");
            maximum = Math.max(maximum, length);
        }
        Preconditions.checkState(layerKvLengths == null, "layer KV widths already configured");
        layerKvLengths = copy;
        kvLength = maximum;
    }

    public int[] layerKvLengths() {
        return layerKvLengths == null ? null : layerKvLengths.clone();
    }

    /** Configures independently-owned cache groups before the first session is opened. */
    public synchronized void configureGroups(List<KvCacheGroupPolicy> policies, int[] layerGroupIndexes) {
        Preconditions.checkState(orderedGroupPolicies.size() == 1
                        && orderedGroupPolicies.get(0).type() == KvCacheGroupType.FULL
                        && this.layerGroupIndexes == null,
                "KV cache groups are already configured");
        Preconditions.checkArgument(policies != null && !policies.isEmpty(), "at least one group is required");
        HashMap<String, KvCacheGroupPolicy> byId = new HashMap<>();
        policies.forEach(policy -> {
            Preconditions.checkArgument(byId.put(policy.groupId(), policy) == null,
                    "duplicate KV cache group: %s", policy.groupId());
        });
        Preconditions.checkArgument(layerGroupIndexes != null && layerGroupIndexes.length == layers,
                "layer group indexes must equal layer count");
        for (int index : layerGroupIndexes) {
            Preconditions.checkArgument(index >= 0 && index < policies.size(), "layer group index out of bounds");
        }
        groupPolicies = Map.copyOf(byId);
        orderedGroupPolicies = List.copyOf(policies);
        this.layerGroupIndexes = layerGroupIndexes.clone();
    }

    public KvCacheSession openSession() {
        metricRegistry.meter("kvcache.v2.session.open").mark();
        if (layerGroupIndexes == null) {
            return openSession("full");
        }
        return new KvCacheSession(layers, contextLength, kvLength, blockSize, dtype, metricRegistry,
                trackReadViews, settings, lighter, layerKvLengths, orderedGroupPolicies,
                layerGroupIndexes);
    }

    public KvCacheSession openSession(String groupId) {
        KvCacheGroupPolicy policy = Objects.requireNonNull(groupPolicies.get(groupId), "unknown group: " + groupId);
        return new KvCacheSession(layers, contextLength, kvLength, blockSize, dtype, metricRegistry,
                trackReadViews, settings, lighter, layerKvLengths, policy);
    }

    /** Opens the independently-owned group selected for a layer. */
    public KvCacheSession openSessionForLayer(int layer) {
        return openSession(groupForLayer(layer).groupId());
    }

    public List<KvCacheGroupPolicy> groupPolicies() {
        return List.copyOf(groupPolicies.values());
    }

    public KvCacheGroupPolicy groupForLayer(int layer) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
        if (layerGroupIndexes == null) return groupPolicies.get("full");
        return orderedGroupPolicies.get(layerGroupIndexes[layer]);
    }

    public KvGroupBlockKey groupBlockKey(String groupId, KvBlockKey blockKey) {
        Preconditions.checkArgument(groupPolicies.containsKey(groupId), "unknown group: %s", groupId);
        return new KvGroupBlockKey(groupId, blockKey.withGroupId(groupId));
    }
}
