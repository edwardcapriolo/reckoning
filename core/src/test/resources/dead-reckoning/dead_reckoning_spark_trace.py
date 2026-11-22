#!/usr/bin/env python3
"""Capture a deterministic Hugging Face Dead Reckoning trace for Spark-X2.6.

This runner is intentionally self-contained. It is executed with the Python
environment that contains Transformers and PyTorch, while the checked-in
launcher supplies that environment from the separate Transformers checkout.
The model directory must already exist locally; this script never downloads a
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import masking_utils
from transformers.modeling_utils import PreTrainedModel


DEFAULT_PROMPT = "What is the capital of Anhui Province?"
EXPECTED_TRANSFORMERS_VERSION = "4.57.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Local Spark checkpoint directory")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL trace")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default=None)
    parser.add_argument("--max-tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--reasoning-effort", choices=("high", "low"), default="high")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument(
        "--allow-compatibility-shims",
        action="store_true",
        help="Allow non-4.57.1 Transformers for exploratory runs; output is not a source baseline",
    )
    return parser.parse_args()


def torch_dtype(value: str):
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def tensor_from_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            tensor = tensor_from_output(value)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for value in output.values():
            tensor = tensor_from_output(value)
            if tensor is not None:
                return tensor
    return None


def tensor_stats(tensor, first_values: int = 32):
    value = tensor.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(value)
    finite_values = value[finite]
    if finite_values.numel():
        minimum = finite_values.min().item()
        maximum = finite_values.max().item()
        total = finite_values.sum().item()
        sum_abs = finite_values.abs().sum().item()
        sum_squares = (finite_values * finite_values).sum().item()
        mean = total / finite_values.numel()
        rms = (sum_squares / finite_values.numel()) ** 0.5
        variance = max(0.0, sum_squares / finite_values.numel() - mean * mean)
        max_abs = finite_values.abs().max().item()
    else:
        minimum = maximum = total = sum_abs = sum_squares = mean = rms = variance = max_abs = float("nan")
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", "").upper(),
        "count": tensor.numel(),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "sum": total,
        "mean_abs": sum_abs / finite_values.numel() if finite_values.numel() else float("nan"),
        "max_abs": max_abs,
        "l2_norm": sum_squares ** 0.5,
        "rms": rms,
        "stddev": variance ** 0.5,
        "zero_count": int((value == 0).sum().item()),
        "nan_count": int(torch.isnan(value).sum().item()),
        "positive_infinity_count": int(torch.isposinf(value).sum().item()),
        "negative_infinity_count": int(torch.isneginf(value).sum().item()),
        "hash": hashlib.sha256(raw).hexdigest(),
        "first_values": value[:first_values].tolist(),
    }


def render_inputs(tokenizer, args: argparse.Namespace):
    messages = []
    if args.system is not None:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    template_kwargs = {"enable_thinking": not args.no_thinking}
    if not args.no_thinking:
        template_kwargs["reasoning_effort"] = args.reasoning_effort
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        **template_kwargs,
    )
    rendered = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
    return encoded, rendered


def install_spark_transformers_compatibility() -> None:
    """Accept Spark's legacy API calls with newer Transformers code."""
    original = PreTrainedModel.get_expanded_tied_weights_keys
    if getattr(original, "_spark_compatibility", False):
        return

    def compatible(self, all_submodels: bool = False):
        if getattr(getattr(self, "config", None), "model_type", None) == "spark2_5":
            tied_keys = getattr(self, "_tied_weights_keys", None)
            if isinstance(tied_keys, list):
                self._tied_weights_keys = {
                    target: "model.embedding.weight" for target in tied_keys
                }
        return original(self, all_submodels=all_submodels)

    compatible._spark_compatibility = True
    PreTrainedModel.get_expanded_tied_weights_keys = compatible

    for function_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original_mask = getattr(masking_utils, function_name)
        if "input_embeds" in inspect.signature(original_mask).parameters:
            continue

        def compatible_mask(*args, _original=original_mask, input_embeds=None, **kwargs):
            if input_embeds is not None:
                kwargs["inputs_embeds"] = input_embeds
            if "cache" not in kwargs and "past_key_values" in kwargs:
                kwargs["cache"] = kwargs.pop("past_key_values")
            # Spark's 4.57 remote code supplies this value, but current mask
            # helpers derive offsets from the cache and no longer accept it.
            kwargs.pop("cache_position", None)
            return _original(*args, **kwargs)

        setattr(masking_utils, function_name, compatible_mask)


def main() -> None:
    args = parse_args()
    model_dir = args.model.expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"model directory has no config.json: {model_dir}")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION and not args.allow_compatibility_shims:
        raise SystemExit(
            f"Spark source baselines require Transformers {EXPECTED_TRANSFORMERS_VERSION}; "
            f"found {transformers.__version__}. Install the pinned version or pass "
            "--allow-compatibility-shims for an explicitly non-authoritative exploratory run."
        )

    if args.allow_compatibility_shims:
        install_spark_transformers_compatibility()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch_dtype(args.dtype),
        trust_remote_code=True,
        local_files_only=True,
    ).to(args.device).eval()
    encoded, rendered_prompt = render_inputs(tokenizer, args)
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    state = {"forward": -1, "phase": "PREFILL", "position": -1}
    records = [{
        "kind": "RUN_START",
        "model": str(model_dir),
        "prompt": rendered_prompt,
        "prompt_tokens": int(encoded["input_ids"].shape[-1]),
        "input_ids": encoded["input_ids"][0].detach().cpu().tolist(),
        "generator_parameters": {
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "do_sample": False,
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "dtype": args.dtype,
            "model_type": getattr(model.config, "model_type", None),
        },
    }]

    def before_model(_module, _inputs):
        state["forward"] += 1
        state["phase"] = "PREFILL" if state["forward"] == 0 else "DECODE"
        state["position"] = (
            -1
            if state["forward"] == 0
            else encoded["input_ids"].shape[-1] + state["forward"] - 1
        )

    hooks = [model.register_forward_pre_hook(before_model)]

    def hook(name):
        def capture(_module, _inputs, output):
            tensor = tensor_from_output(output)
            if tensor is None:
                return
            record = {
                "kind": "CHECKPOINT",
                "phase": state["phase"],
                "position": state["position"],
                "name": name,
            }
            record.update(tensor_stats(tensor))
            records.append(record)

        return capture

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(hook(name)))

    try:
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_tokens,
                do_sample=False,
            )
    finally:
        for handle in hooks:
            handle.remove()

    generated_tokens = generated[0, encoded["input_ids"].shape[-1]:].tolist()
    records.append({
        "kind": "RESPONSE_END",
        "text": tokenizer.decode(generated_tokens, skip_special_tokens=False),
        "generated_tokens": generated_tokens,
        "finish_reason": "MAX_TOKENS",
    })
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=True) + "\n")

    print(f"HF_DEAD_RECKONING_TRACE={args.output.resolve()}")
    print(f"HF_DEAD_RECKONING_RESPONSE={records[-1]['text']!r}")
    print(f"HF_DEAD_RECKONING_TOKENS={generated_tokens}")


if __name__ == "__main__":
    main()
