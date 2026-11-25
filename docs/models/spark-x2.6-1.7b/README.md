# Spark-X2.6-1.7B

## Selection

- Checkpoint: `XHToken/Spark-X2.6-1.7B`
- Revision: `0e60633b345b19ba9f56ed1fda197acd271e8546`
- Base model: `XHToken/Spark-X2.6-1.7B-Base`
- Variant: instruction-tuned, reasoning-capable causal language model
- License: Apache-2.0
- Architecture: `Spark2_5ForCausalLM`
- Config model type: `spark2_5`
- Transformers version in config: `4.57.1`
- Parameters: `1,707,657,216` BF16 parameters
- Checkpoint storage: `3,415,314,432` bytes across two safetensors shards

This is the selected first Spark target because it is the exact model requested for the 1M-token context use case. The model card states that Spark-X2.6-1.7B retains the Spark-X2.5 architecture; the revision-specific source files are the implementation authority.

## Architecture Summary

- 28 decoder layers with a repeating 3-sliding / 1-full attention pattern.
- Hidden size `2048`, intermediate size `6656`, and head dimension `256`.
- Grouped-query attention with 8 query heads and 2 key/value heads.
- Sliding attention window `512`.
- Sliding layers use full-width RoPE with theta `10000`.
- Full-attention layers use 25% partial RoPE with theta `5000000`.
- Attention output is gated per query head through a sigmoid projection.
- MLP uses GELU gated activation: `down(gelu(gate(x)) * up(x))`.
- Input embeddings are tied to the output projection.
- The model uses BF16 weights while the upstream forward path performs selected normalization, RoPE, and softmax work in FP32.

## Context And Prompting

The model config declares `max_position_embeddings=1048576` and the generation config permits the same token budget. The tokenizer metadata declares `model_max_length=131072`, which is a material metadata mismatch requiring explicit handling during implementation and validation.

The shipped chat template defaults to thinking enabled and supports `reasoning_effort=high|low`, or thinking disabled. It emits Spark-specific role markers and `<think>` boundaries, so tokenizer/chat-template parity is part of the port rather than optional presentation behavior.

## References

- Model card: https://huggingface.co/XHToken/Spark-X2.6-1.7B
- Exact revision: https://huggingface.co/XHToken/Spark-X2.6-1.7B/tree/0e60633b345b19ba9f56ed1fda197acd271e8546
- Config: https://huggingface.co/XHToken/Spark-X2.6-1.7B/raw/0e60633b345b19ba9f56ed1fda197acd271e8546/config.json
- Upstream config implementation: https://huggingface.co/XHToken/Spark-X2.6-1.7B/raw/0e60633b345b19ba9f56ed1fda197acd271e8546/configuration_spark.py
- Upstream model implementation: https://huggingface.co/XHToken/Spark-X2.6-1.7B/raw/0e60633b345b19ba9f56ed1fda197acd271e8546/modeling_spark.py
- Checkpoint index: https://huggingface.co/XHToken/Spark-X2.6-1.7B/raw/0e60633b345b19ba9f56ed1fda197acd271e8546/model.safetensors.index.json
- Chat template: https://huggingface.co/XHToken/Spark-X2.6-1.7B/raw/0e60633b345b19ba9f56ed1fda197acd271e8546/chat_template.jinja

No existing Reckoning model implementation is a correctness reference for this port. In particular, Gemma 4 behavior is not used as evidence because it has not reliably produced clean data.
