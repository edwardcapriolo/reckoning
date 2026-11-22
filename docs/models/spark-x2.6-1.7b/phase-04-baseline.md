# Phase 4: Dead Reckoning Baseline

## Status

Source baseline captured for operator review. Phase 5 is not started.

## Checked-In Runner

The Python source reference runner is:

```text
core/src/test/resources/dead-reckoning/dead_reckoning_spark_trace.py
```

It is self-contained and does not import the untracked scripts in the separate Transformers checkout. It loads the local model with `trust_remote_code=True` and `local_files_only=True`, applies Spark's chat template, uses deterministic generation (`do_sample=False`, temperature `0`, fixed seed), captures module checkpoints, and emits the shared JSONL `RUN_START`, `CHECKPOINT`, and `RESPONSE_END` record types.

The model declares Transformers `4.57.1` as its source version. The runner rejects other versions by default. An explicit `--allow-compatibility-shims` option exists for exploratory diagnosis only; it translates known API differences in memory and its output must not be used as a source baseline.

The runner also contains a narrow compatibility shim for the current Transformers checkout: Spark's remote class declares `_tied_weights_keys` in the older list form, while the current Transformers API requires a target-to-source mapping. The shim converts only `spark2_5` to `lm_head.weight -> model.embedding.weight` in memory.

The launcher is:

```text
scripts/run_spark_dead_reckoning.sh
```

It manages a dedicated repository-local environment at `.venv-spark-dead-reckoning` by default. `--setup` creates or updates it from `scripts/spark-dead-reckoning-requirements.txt`; `--venv` can select another isolated environment. It accepts a local model through `--model` and exposes output, device, dtype, prompt, thinking, reasoning-effort, and token-limit options.

Example:

```bash
scripts/run_spark_dead_reckoning.sh \
  --setup \
  --model /home/ecapriolo/.reckoning/XHToken_Spark-X2.6-1.7B \
  --output target/dead-reckoning-hf/spark-x2.6-1.7b.jsonl \
  --max-tokens 6
```

## Validation Performed

- Python syntax validation passed with `python3 -m py_compile`.
- Shell syntax validation passed with `bash -n`.
- Launcher help and argument rendering passed.
- The pinned requirements are checked in at `scripts/spark-dead-reckoning-requirements.txt`.

An exploratory run against the newer Transformers checkout loaded all 226 tensors but generated `WeWeWeWeWeWe`. That output is rejected as a baseline and demonstrated why the source environment must be pinned to `4.57.1`.

The dedicated repository-local environment then loaded both checkpoint shards successfully and produced:

```text
HF_DEAD_RECKONING_TRACE=target/dead-reckoning-hf/spark-x2.6-1.7b.jsonl
HF_DEAD_RECKONING_TOKENS=[3066, 1633, 6787, 27, 11340, 27179]
HF_DEAD_RECKONING_RESPONSE='We need answer. User asks'
```

The six-token response ends inside Spark's thinking channel and is intentionally not treated as a completed-answer quality oracle. The source evidence is the rendered prompt, input IDs, checkpoint records, runtime metadata, and selected token sequence.

## Current Blocker

The earlier local cache inspected in the Linux workspace contained only:

- `model-00001-of-00002.safetensors`
- metadata files including `config.json`

It did not contain `model-00002-of-00002.safetensors` and had no `.finished` marker. The authoritative run was later completed in the dedicated macOS venv.

## Trace Contract

The Python trace is an HF source trace, not a golden output assertion. Once generated, compare it with a Reckoning trace using `DeadReckoningCrossRuntimeCheck` after adding Spark-specific stage mappings. The current generic mappings do not understand Spark's fused `q_k_v_proj` or `g_proj` names, so a direct GREEN result is not expected until those mappings and the Java checkpoint names are intentionally aligned.

## Gate

Phase 4 is complete for operator review. The next step is to inspect the trace and begin Phase 5 characterization tests; no model implementation has been added.
