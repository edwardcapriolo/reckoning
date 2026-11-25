# Config Field Analysis

Source: `XHToken/Spark-X2.6-1.7B/config.json`, revision `0e60633b345b19ba9f56ed1fda197acd271e8546`.

| Field | JSON type | Value | Reckoning status | Required behavior / evidence |
| --- | --- | --- | --- | --- |
| `architectures` | array<string> | `Spark2_5ForCausalLM` | new dispatch | Select the causal-LM root class. |
| `attention_bias` | boolean | `false` | understood, new family | Q/K/V, gate, and output projections have no bias. |
| `attention_dropout` | number | `0.0` | understood | Dropout is inactive for inference. |
| `auto_map` | object | custom Spark config/model classes | source metadata | Identifies the upstream remote-code implementation. |
| `bos_token_id` | integer | `0` | known | BOS token metadata. |
| `dtype` | string | `bfloat16` | known dtype metadata | Dense checkpoint weights are BF16. |
| `eos_token_id` | integer | `1` | known | EOS token metadata. |
| `gate_attn_act_mode` | string | `sigmoid` | new behavior | Apply sigmoid to the per-head attention gate. |
| `head_dim` | integer | `256` | known | Query and KV head width. |
| `headwise_attn_output_gate` | boolean | `true` | new behavior | Project one gate value per query head and multiply attention output before `out_proj`. |
| `hidden_act` | string | `gelu` | known activation | MLP uses GELU on `gate_proj(x)`. |
| `hidden_size` | integer | `2048` | known | Decoder and embedding width. |
| `initializer_range` | number | `0.0221` | training metadata | Relevant for construction only; not used when loading the checkpoint. |
| `intermediate_size` | integer | `6656` | known | MLP intermediate width. |
| `layer_types` | array<string> | 21 `sliding_attention`, 7 `full_attention` | existing concept, Spark schedule required | Select mask and RoPE by layer; sequence is 3 sliding layers followed by 1 full layer. |
| `max_position_embeddings` | integer | `1048576` | known, high risk | Must support positions through the 1M-token limit without full-context RoPE preallocation. |
| `mlp_bias` | boolean | `false` | understood, new family | MLP projections have no bias. |
| `model_type` | string | `spark2_5` | new dispatch | Register config/model selection. |
| `num_attention_heads` | integer | `8` | known | Query head count. |
| `num_hidden_layers` | integer | `28` | known | Decoder depth. |
| `num_key_value_heads` | integer | `2` | known GQA | Repeat each KV head across four query heads for attention. |
| `pad_token_id` | integer | `2` | known | Padding metadata and embedding padding index. |
| `rms_norm_eps` | number | `1e-6` | known | RMSNorm epsilon. |
| `rope_parameters.sliding_attention.partial_rotary_factor` | number | `1.0` | existing concept | Rotate the complete 256-dimensional head for sliding layers. |
| `rope_parameters.sliding_attention.rope_theta` | integer | `10000` | existing concept | Sliding-layer inverse-frequency base. |
| `rope_parameters.full_attention.partial_rotary_factor` | number | `0.25` | existing concept, exact source mapping required | Rotate the first 64 dimensions and leave the remaining dimensions unchanged. |
| `rope_parameters.full_attention.rope_theta` | integer | `5000000` | existing concept | Full-layer inverse-frequency base. |
| `tie_word_embeddings` | boolean | `true` | known, required | Compute logits with the input embedding matrix. |
| `transformers_version` | string | `4.57.1` | provenance | Pins the source API behavior reviewed for this phase. |
| `use_cache` | boolean | `true` | known, high risk | Preserve incremental KV state across generation. |
| `vocab_size` | integer | `131072` | known | Embedding and logits vocabulary width. |

## Related Metadata Outside `config.json`

| File / field | Value | Status |
| --- | --- | --- |
| `generation_config.max_tokens` | `1048576` | Confirms the intended generation ceiling. |
| `generation_config.temperature` | `1.0` | Source default; runtime sampling policy. |
| `generation_config.top_p` | `0.95` | Source default; runtime sampling policy. |
| `tokenizer_config.model_max_length` | `131072` | Mismatch with model context; must be resolved explicitly, not silently assumed equivalent. |
| `tokenizer_config.add_bos_token` | `false` | Tokenizer behavior. |
| `tokenizer_config.add_eos_token` | `false` | Tokenizer behavior. |
| `tokenizer_config.tokenizer_class` | `PreTrainedTokenizerFast` | Grace tokenizer compatibility investigation required. |
| `tokenizer_config.chat_template` | Spark Jinja template | Must preserve role markers, tools, and thinking controls. |

## Checkpoint Metadata

- `total_parameters`: `1707657216`
- `total_size`: `3415314432` bytes
- Shards: `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`
- Weight naming uses `model.embedding`, `model.layers.<n>`, and `model.norm`.
- Each layer contains `self_attn.q_k_v_proj`, `self_attn.g_proj`, `self_attn.out_proj`, three MLP projections, and two layer norms.
- The index maps the tied output embedding through the input embedding; no separate `lm_head.weight` tensor is present.
