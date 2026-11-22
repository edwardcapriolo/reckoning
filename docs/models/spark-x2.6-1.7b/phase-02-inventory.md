# Phase 2: File, Config, And Weight Inventory

## Status

Complete for operator review. Phase 3 is not started.

## Inventory Test

The inventory is implemented by:

```text
core/src/test/java/io/teknek/reckoning/model/spark/Spark2_5InventoryIT.java
```

The test uses `ModelFetcher` with `FetchPolicy.FULL_MODEL`, reads the downloaded configuration through `Spark2_5Config`, and opens the safetensors headers with `DefaultWeightLoader`. It does not load full tensor values into memory or instantiate a model.

The successful invocation used the repository-local toolchain file and workspace-local Jansi directory:

```bash
MAVEN_OPTS="-XX:TieredStopAtLevel=1 -Djansi.tmpdir=/ai-code/del2/.jansi" \
mvn -Dmaven.toolchains.file=/tmp/opencode/spark-toolchains.xml \
  -pl core \
  -Dit.test=Spark2_5InventoryIT \
  -DexcludedITGroups=none \
  -Dfailsafe.failIfNoSpecifiedTests=false \
  failsafe:integration-test failsafe:verify
```

Result: passed after correcting the expected tensor count from 198 to 226. The initial mismatch was a test arithmetic error: each layer has 8 tensors, not 7.

## Repository Contents

The complete fetch contains the expected model artifacts:

- `config.json`
- `configuration_spark.py`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `model.safetensors.index.json`
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- `.finished`

The selected checkpoint has two safetensors shards and the index reports `1,707,657,216` parameters and `3,415,314,432` bytes of tensor data.

## Parsed Configuration

The typed config successfully verifies:

- `spark2_5` and `Spark2_5ForCausalLM` identity.
- 28 layers, hidden size 2048, intermediate size 6656.
- 8 query heads, 2 KV heads, and head dimension 256.
- 1,048,576 maximum positions and a 512-token sliding window.
- Vocabulary size 131,072.
- Full-attention rotary dimension 64 with theta 5,000,000.
- Sliding-attention rotary dimension 256 with theta 10,000.

## Tensor Inventory

The loader discovers exactly 226 tensors, all BF16:

- 1 input embedding tensor: `model.embedding.weight`, shape `[131072, 2048]`.
- 1 final norm tensor: `model.norm.weight`, shape `[2048]`.
- 28 layers with 8 tensors each:
  - `input_layernorm.weight`, `[2048]`
  - `post_attention_layernorm.weight`, `[2048]`
  - `self_attn.q_k_v_proj.weight`, `[3072, 2048]`
  - `self_attn.g_proj.weight`, `[8, 2048]`
  - `self_attn.out_proj.weight`, `[2048, 2048]`
  - `mlp.gate_proj.weight`, `[6656, 2048]`
  - `mlp.up_proj.weight`, `[6656, 2048]`
  - `mlp.down_proj.weight`, `[2048, 6656]`

No separate `lm_head.weight` exists, confirming that the output projection must use tied input embeddings.

## Unknowns Remaining

- The index and headers establish topology and dtype, but not forward numerical parity.
- The tokenizer JSON and chat template have been fetched but not yet behaviorally characterized.
- The inventory does not establish whether the current KV cache can implement Spark's mixed sliding/full attention schedule.
- No quantization sidecar or alternate weight layout is present in this checkpoint.

## Phase 2 Gate

The exact checkpoint is fetchable and readable by Reckoning's safetensors loader. Its typed configuration and real tensor topology are recorded. Phase 3 may begin only after operator approval.
