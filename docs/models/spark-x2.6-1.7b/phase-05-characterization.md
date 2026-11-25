# Phase 5: Characterization Tests

## Status

Phase 5 is closed for operator review with a deliberately limited scope. The tests cover concrete reusable/runtime boundaries; abstract attention/cache contracts are deferred to Phase 6 implementation rather than represented by detached tests.

## First Component: GELU-Gated MLP

Test:

```text
core/src/test/java/io/teknek/reckoning/model/spark/Spark2_5MlpCharacterizationTest.java
```

The fixture represents a real Spark operation with a deliberately tiny width and uses a minimal `Spark2_5Config` plus the existing tensor-native `MLPBlock2` with a Mockito `AbstractModel` shell:

- Hidden size: `2`
- Intermediate size: `1`
- Input: `[1.0, 2.0]`
- Gate projection: `[1.0, 0.0]`
- Up projection: `[2.0, 0.0]`
- Down projection: `[3.0]`

The contract is:

```text
down(gelu(gate(input)) * up(input))
```

The independent fixture calculation produces expected output `5.048068` for the first output dimension.

## Test Result

Command:

```bash
MAVEN_OPTS="-XX:TieredStopAtLevel=1 -Djansi.tmpdir=/ai-code/del2/.jansi" \
mvn -Dmaven.toolchains.file=/tmp/opencode/spark-toolchains.xml \
  -pl core -Dtest=Spark2_5MlpCharacterizationTest test
```

The initial red gate was:

```text
RED characterization gate: implement Spark2_5 MLP before asserting expected=5.048068
```

After wiring real `TensorRef` fixtures and a mocked model shell, the same test passed: one test executed, zero failures, zero errors. The test uses a real `Spark2_5Config`, `Lighter`, `CompositeOps`, and `MLPBlock2`; only model services such as tensor allocation, tensor-parallel context, and synchronous chunk execution are mocked.

## Existing Related Work

`MLPBlock2` already contains a production SwiGLU-style path with configurable activation, gate/up multiplication, down projection, chunking, quantization, and LoRA hooks. `EmbeddingGemma2MlpOperation` provides an isolated operation pattern, but uses `GELU_PYTORCH_TANH`; Spark declares `hidden_act=gelu`, so it is not a direct Spark oracle.

## Decision

Spark can initially reuse `MLPBlock2`: its gate/up/multiply/down ordering is compatible, and the mocked config selects `ActivationFunction.Type.GELU`. No new Spark-specific MLP production class was required for this characterization.

The remaining question is whether the fused activation/quantization path preserves the same result for Spark's real BF16 weights and production provider choices. That requires a follow-up provider/dtype characterization before full integration.

## Layer-Specific RoPE

The first genuinely Spark-specific component is implemented and tested by:

```text
core/src/main/java/io/teknek/reckoning/model/spark/Spark2_5Rope.java
core/src/test/java/io/teknek/reckoning/model/spark/Spark2_5RopeTest.java
```

`Spark2_5Rope` reuses the existing on-demand `CompositeOps` rotation and constructs layer-type-specific inverse frequencies. The test uses the real 256-dimensional head shape and verifies that sliding attention rotates all dimensions while full attention rotates only the first 64 and passes the remaining dimensions through unchanged.

## Additional Concrete Coverage

- `Spark2_5AttentionLayout` and `Spark2_5AttentionLayoutTest` cover fused QKV split shapes and GQA repetition from 2 KV heads to 8 query heads.
- `OutputHeadPolicyTest` covers tied and untied output-weight selection and actual logits projection with deterministic tensors.
- `LocalSparkPromptTemplateTest` renders high- and low-effort thinking prompts from the cached Spark template.

## Deferred To Phase 6

- Provider-backed per-head sigmoid attention gating.
- Actual mixed sliding/full attention mask integration.
- Mixed-layer KV-cache prefill/decode behavior.
- Spark model output-head loading and ownership integration.
- Grace-versus-Transformers token ID parity.

These items require the actual Spark attention/model boundaries. Detached tests would not provide useful evidence.

## Phase Gate

Phase 5 is complete at this scope. Phase 6 may begin with these deferred items recorded as implementation and validation risks.

## Focused Suite Evidence

Tensor/provider characterization command:

```bash
mvn -Dmaven.toolchains.file=/tmp/opencode/spark-toolchains.xml \
  -pl tensor -am \
  -Dtest=HeadwiseSigmoidGateFuzzTest,PanamaSaxpyDTypeTest,CompositeOpsRotaryEmbeddingTest \
  -Dsurefire.failIfNoSpecifiedTests=false test
```

Result: 6 tests passed, 0 failures, 0 errors.

Spark/core characterization command:

```bash
mvn -Dmaven.toolchains.file=/tmp/opencode/spark-toolchains.xml \
  -pl core -am \
  -Dtest=Spark2_5MlpCharacterizationTest,Spark2_5TiedEmbeddingConfigTest,\
Spark2_5RopeTest,Spark2_5AttentionLayoutTest,OutputHeadPolicyTest,\
LocalSparkPromptTemplateTest \
  -Dsurefire.failIfNoSpecifiedTests=false test
```

Result: 7 tests passed, 0 failures, 0 errors. Maven emitted known environment toolchain-path warnings but the reactor build completed successfully.
