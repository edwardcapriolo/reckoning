package io.teknek.reckoning.tensor.kv;

import java.util.Objects;

/** Group-scoped identity used before a group-aware prefix cache is introduced. */
public record KvGroupBlockKey(String groupId, KvBlockKey blockKey) {
    public KvGroupBlockKey {
        groupId = Objects.requireNonNull(groupId, "groupId");
        blockKey = Objects.requireNonNull(blockKey, "blockKey");
    }
}
