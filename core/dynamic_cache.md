# Spark Dynamic Cache Port Gaps

The Spark Transformers implementation creates its cache with:

```python
DynamicCache(config=self.config)
```

This creates `DynamicLayer` instances for full-attention layers and
`DynamicSlidingWindowLayer` instances for sliding-attention layers.

## Confirmed Gaps

### 1. Sliding-layer cache is never truncated

Transformers retains only the latest `sliding_window - 1` cached rows for a
sliding layer:

```python
self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
```

For Spark, the window is `512`, so the persistent sliding cache retains `511`
rows and the current rows are supplied separately for the current attention
operation.

Reckoning's `KvCacheSession` stores every row for every Spark layer. This
preserves masked attention results, but differs in persistent cache contents,
memory use, and cache behavior.

### 2. Cache storage is not layer-specific

Transformers applies a separate cache-layer implementation and retention policy
for each `layer_idx`. Reckoning has one `KvCacheSession` with layer indexing,
but no Spark-specific per-layer retention policy.

### 3. `cache_position` is reconstructed, not passed through

Transformers passes explicit cache positions:

```python
cache_position
```

Reckoning reconstructs them as:

```java
startPosition + row
```

This matches normal contiguous generation, but does not faithfully support
non-contiguous cache positions or alternate cache callers.

### 4. The source update return value is implicit in Java

Transformers returns the updated states directly:

```python
k, v = past_key_values.update(...)
```

Those returned tensors are then passed directly into attention.

Reckoning writes KV rows, then separately creates a `KvReadView` and combines
it with current keys and values for attention. For normal contiguous generation
this is output-equivalent, but the source operation boundary and returned-state
semantics are not represented directly.

### 5. Sliding-cache returned-state behavior is not modeled

For a sliding layer, Transformers stores only the recent cache internally but
returns the full relevant concatenation for the current attention call.

Reckoning has no distinction between:

- persistent cached state
- attention-visible returned state

It uses the full persistent cache plus a later attention window.

## Not Gaps

- `sin` and `cos` are not missing cache behavior for this model's default
  `DynamicCache`; the dynamic layer implementations ignore them.
- Per-row Java writes are inefficient, but are not a fidelity gap by
  themselves.
- The layer index mapping is correct: `layerIndex` corresponds to
  `self.layer_idx`.

The primary confirmed gap is the missing `DynamicSlidingWindowLayer` behavior.
