# Phase 3: Source And Test Analysis

## Status

Complete for operator review. Phase 4 is not started.

## Source Authority

The primary source is the exact model revision:

- `configuration_spark.py`: `Spark2_5Config`
- `modeling_spark.py`: `Spark2_5ForCausalLM`, `Spark2_5Model`, decoder, attention, MLP, RMSNorm, and helper functions
- `tokenizer_config.json` / `chat_template.jinja`: tokenizer and prompt contract

The source revision is `0e60633b345b19ba9f56ed1fda197acd271e8546` and the model declares Transformers `4.57.1`.

The current public Transformers repository does not expose `spark2_5` under the expected `src/transformers/models/` or `tests/models/` paths. Therefore, no official upstream Spark test file was available to port directly. The XHToken vLLM plugin is supplemental evidence only. It independently confirms the fused projection, per-head sigmoid gate, GELU-and-multiply MLP, mixed attention configuration, and fixed-history attention test shape, but it is not the behavioral authority over the selected Hugging Face checkpoint.

Supplemental source:

- https://github.com/XHToken/Spark-plugin/tree/main
- `src/vllm_spark2_5_plugin/spark2_5.py`
- `src/vllm_spark2_5_plugin/spark2_5_config.py`
- `src/vllm_spark2_5_plugin/spark_triton_attention.py`
- `tests/test_smoke.py`
- `tests/test_spark_triton_attention.py`

## Source-Test Matrix

| Source implementation or behavior | Source assertion / invariant | Target test or evidence | Status |
| --- | --- | --- | --- |
| `Spark2_5Config.__init__` | Defaults and validation for dimensions, GQA, layer types, sliding window, and per-type RoPE | `Spark2_5InventoryIT` parses the real config and asserts all real values | complete for inventory |
| `compute_rope_cos_sin` | Uses current positions, layer-specific theta, and `head_dim * partial_rotary_factor`; returns matching cos/sin widths | New focused RoPE characterization test using independent reference calculations | required before Phase 6 |
| `apply_rotary_pos_emb` | Rotates only the configured prefix; passes remaining dimensions unchanged; computes in FP32 and returns input dtype | New focused partial/full rotation value test | required before Phase 6 |
| `repeat_kv` | Repeats 2 KV heads into 8 query-head slots with four-way grouping | New GQA layout test with distinct KV-head values | required before Phase 6 |
| `eager_attention_forward` | Scales by `1/sqrt(head_dim)`, applies mask, stabilizes softmax, computes in FP32, and multiplies values | New attention oracle test for masked and unmasked rows | required before Phase 6 |
| `Spark2_5RMSNorm.forward` | Variance is mean squared value; normalization and learned weight multiply occur in FP32; output returns input dtype | New RMSNorm value/dtype test | required before Phase 6 |
| `Spark2_5MLP.forward` | Exact order is `down_proj(GELU(gate_proj(x)) * up_proj(x))`; all projections are bias-free | New MLP order test with deterministic small weights | required before Phase 6 |
| `Spark2_5Attention.__init__` and `forward` | Splits fused QKV as Q, K, V; shapes Q as 8 heads and K/V as 2 heads; applies RoPE before cache update; gates attention per head before output projection | New attention mechanics test with representative real projection shapes | required before Phase 6 |
| `gate_attn_act_mode` | `sigmoid` is applied in FP32; unsupported modes fail explicitly; gate projection remains BF16-compatible | New gate activation test for sigmoid and unsupported mode | required before Phase 6 |
| `Spark2_5DecoderLayer.forward` | Order is input norm, attention, residual, post-attention norm, MLP, residual | New decoder block operation-order test | required before Phase 6 |
| `Spark2_5Model.forward` | Builds full and sliding causal masks, tracks cache positions, computes per-layer-type RoPE, runs all layers, and applies final norm | New small deterministic model-forward characterization test | required before Phase 6 |
| `Spark2_5ForCausalLM.forward` | Tied embeddings are used for logits when configured; `logits_to_keep` slices final hidden states | New tied-logits shape/value test | required before Phase 6 |
| `DynamicCache` integration | Cache update receives current sin/cos and cache positions; full and sliding layers must preserve their own visibility semantics | New cache prefill/decode and mixed-layer mask test | blocking for Phase 6 |
| `tokenizer_config.json` and chat template | No automatic BOS/EOS; Spark role markers; default thinking; high/low effort; tools; `<think>` delimiters | Grace tokenizer fixture and rendered-prompt assertions for user, assistant, tool, and thinking-disabled cases | required before Phase 6 |
| XHToken vLLM plugin fixed-history test | A 512-token history window is retained for sliding attention; ragged requests and GQA are tested | Supplemental reference for later cache/attention tests, not a direct target oracle | reviewed |

## Operation Order Notes

The source performs these boundaries explicitly:

1. Embeddings are created, then the model hidden state is promoted to FP32 before decoder execution.
2. Each decoder layer normalizes and casts to the projection weight dtype before attention.
3. Fused Q/K/V projection is split and reshaped before RoPE.
4. RoPE is applied before cache update and attention.
5. Attention softmax is stabilized and calculated in FP32.
6. Headwise sigmoid gating occurs after attention value aggregation and before head merge/output projection.
7. The post-attention norm and MLP repeat the source's normalization/cast boundary.
8. Final norm returns to the embedding dtype; tied logits use the embedding matrix directly.

These are source semantics, not assumptions derived from another Reckoning model.

## Python Inventory

There are no Python files directly in the repository root. The repository's Python files are:

| Path | Role | Spark relevance |
| --- | --- | --- |
| `scripts/upload_hf_model.py` | Uploads a local directory to Hugging Face using `HF_TOKEN` | operational only; not model execution |
| `core/scripts/sentence_transformers_embedding_showdown.py` | Generates Sentence Transformers embedding vectors and timing reports | unrelated EmbeddingGemma benchmark |
| `core/scripts/compare_embedding_showdown.py` | Compares Java/Python embedding vectors using error, cosine, and quantized hashes | unrelated embedding parity tool |
| `core/scripts/embedding_gemma2_reference.py` | Emits one deterministic EmbeddingGemma 2 Python reference vector | unrelated reference oracle |
| `grace/scripts/hf_fast_tokenizer_benchmark.py` | Benchmarks Hugging Face `tokenizers` throughput and hashes token IDs | reusable methodology only; not Spark-specific |
| `core/src/test/resources/granite_moe_hybrid_reference.py` | Runs a local Transformers Granite checkpoint and prints reference logits | Granite-only oracle pattern |
| `core/src/test/resources/dead-reckoning/dead_reckoning_granite_trace.py` | Captures local Hugging Face Granite tensor hooks and generation trace | Granite-only Dead Reckoning oracle pattern |
| `core/src/test/resources/a.py` | Small manual Gemma 4 chat-template experiment | experimental scratch file; not Spark-related |

The Python files do not form a hidden model runtime or build system for Reckoning. They are scripts and test resources for model upload, embedding/tokenizer benchmarks, or independent Hugging Face references. Spark port validation can later add a Spark-specific reference runner, but none exists today.

## Phase 3 Gate

The Spark source operation order, real checkpoint topology, prompt contract, supplemental vLLM evidence, and repository Python roles are documented. No Spark forward-pass implementation has been added. Phase 4 may begin only after operator approval.
