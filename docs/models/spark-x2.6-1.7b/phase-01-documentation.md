# Phase 1: Documentation And Model Selection

## Status

Complete for operator review. Phase 2 is not started.

## Selected Checkpoint

`XHToken/Spark-X2.6-1.7B` at revision `0e60633b345b19ba9f56ed1fda197acd271e8546`.

This instruction-tuned checkpoint is selected instead of the base checkpoint because the requested target is the usable chat/reasoning model and its shipped chat template exercises the intended thinking controls. The base model is recorded as the upstream base, not as the implementation target.

## Config Summary

- Architecture: `Spark2_5ForCausalLM`
- Model type: `spark2_5`
- 28 decoder layers
- Hidden size: `2048`
- Intermediate size: `6656`
- Attention heads: `8`
- KV heads: `2`
- Head dimension: `256`
- Context: `1048576`
- Sliding window: `512`
- Vocabulary: `131072`
- Configured dtype: BF16
- Parameters: `1,707,657,216`

## Source Findings

The upstream source defines a decoder-only architecture with RMSNorm, GELU gated MLP, fused Q/K/V projection, optional per-head attention gating, GQA KV repetition, layer-type-specific causal masks, and layer-type-specific RoPE. Full-attention layers use the same KV head count as sliding layers; their distinction is the mask and RoPE configuration, not a separate projection width.

The source computes RoPE for the current `cache_position` range rather than constructing a table for the complete configured context. This is aligned with the existing on-demand RoPE motivation in Reckoning and is important for the 1M-token target.

## Existing Reckoning Mapping

No `spark2_5` model or config dispatch exists. No existing model implementation is treated as a correctness reference for Spark. The upstream Spark source and later independent characterization tests are the behavioral authorities for layer types, attention masks, RoPE, projection layout, gating, cache behavior, and dtype boundaries.

The required new behavior is the Spark-specific fused projection layout, headwise sigmoid gate, exact GELU MLP ordering, tied output embedding, Spark tokenizer/chat template behavior, and 1M-context cache/mask behavior.

## Blocking And High-Risk Questions

- Whether the current Reckoning attention/cache path can represent a 512-token sliding window while preserving full-attention layers and incremental positions.
- Whether all tensor providers support the fused projection and per-head gate without scalar production fallbacks.
- How Grace should represent the tokenizer metadata mismatch: tokenizer `model_max_length=131072` versus model context `1048576`.
- Exact mask and cache behavior for a full-attention layer after prior sliding layers.
- Exact BF16-to-FP32 boundaries for RMSNorm, RoPE, softmax, and gate activation.
- Whether the current prompt API can render the shipped Jinja behavior, including `reasoning_effort`, tools, and `<think>` markers.

These are implementation and validation gates for later phases, not reasons to change the selected checkpoint.

## Source Material Reviewed

- Model card and quickstart at the selected revision.
- `config.json`.
- `configuration_spark.py`.
- `modeling_spark.py`.
- `generation_config.json`.
- `tokenizer_config.json` and `chat_template.jinja` metadata.
- `model.safetensors.index.json`.
- Repository model-dispatch and loader structure was checked only to establish that Spark support is absent; no existing model implementation was used as behavioral evidence.

## Phase 1 Gate

Phase 1 is complete for operator review. The next phase is a Java-based repository and weight inventory using Reckoning loaders. No model implementation or test code has been added.
