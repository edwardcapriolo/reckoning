package io.teknek.reckoning.tensor.kv;

import com.google.common.base.Preconditions;

import java.util.Objects;

/** Immutable policy describing one independently-owned KVCache2 group. */
public record KvCacheGroupPolicy(String groupId, KvCacheGroupType type, int window) {
    public KvCacheGroupPolicy {
        groupId = Objects.requireNonNull(groupId, "groupId");
        Preconditions.checkArgument(!groupId.isBlank(), "groupId must not be blank");
        type = Objects.requireNonNull(type, "type");
        Preconditions.checkArgument(type == KvCacheGroupType.FULL ? window == 0 : window > 0,
                "FULL groups require window 0 and SLIDING groups require a positive window");
    }

    public static KvCacheGroupPolicy full() {
        return new KvCacheGroupPolicy("full", KvCacheGroupType.FULL, 0);
    }

    public static KvCacheGroupPolicy full(String groupId) {
        return new KvCacheGroupPolicy(groupId, KvCacheGroupType.FULL, 0);
    }

    public static KvCacheGroupPolicy sliding(String groupId, int window) {
        return new KvCacheGroupPolicy(groupId, KvCacheGroupType.SLIDING, window);
    }

    /** Absolute position of the first persistent row at the supplied logical length. */
    public int persistentStart(int length) {
        Preconditions.checkArgument(length >= 0, "length must be >= 0");
        return type == KvCacheGroupType.FULL ? 0 : Math.max(0, length - window + 1);
    }

    public int persistentTokenCount(int length) {
        return length - persistentStart(length);
    }

    public int attentionStart(int length, int currentRows) {
        Preconditions.checkArgument(currentRows >= 0 && currentRows <= length,
                "currentRows out of bounds");
        return persistentStart(length);
    }

    public int attentionEndExclusive(int length, int currentRows) {
        Preconditions.checkArgument(currentRows >= 0, "currentRows must be >= 0");
        return length + currentRows;
    }
}
