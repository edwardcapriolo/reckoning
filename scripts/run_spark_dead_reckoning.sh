#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_spark_dead_reckoning.sh --model DIR [options]

Options:
  --venv DIR       Python virtual environment (default: .venv-spark-dead-reckoning)
  --setup          Create/update the selected venv from the pinned requirements
  --model DIR      Local Spark-X2.6 checkpoint directory
  --output FILE    JSONL output (default: target/dead-reckoning-hf/spark-x2.6-1.7b.jsonl)
  --device DEVICE  PyTorch device (default: cpu)
  --dtype DTYPE    auto, float32, float16, or bfloat16 (default: bfloat16)
  --max-tokens N   Deterministic generation budget (default: 6)
  --prompt TEXT    User prompt
  --no-thinking    Disable Spark thinking in the chat template
  --reasoning-effort VALUE  high or low (default: high)
  --allow-compatibility-shims  Permit non-4.57.1 exploratory runs (not a baseline)
EOF
  exit "${1:-2}"
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv_dir=${SPARK_DEAD_RECKONING_VENV:-$repo_root/.venv-spark-dead-reckoning}
setup=0
model_dir=
output="$repo_root/target/dead-reckoning-hf/spark-x2.6-1.7b.jsonl"
device=cpu
dtype=bfloat16
max_tokens=6
prompt=
no_thinking=0
reasoning_effort=high
allow_compatibility_shims=0

while (($#)); do
  case "$1" in
    --venv) [[ $# -ge 2 ]] || usage; venv_dir=$2; shift 2 ;;
    --setup) setup=1; shift ;;
    --model) [[ $# -ge 2 ]] || usage; model_dir=$2; shift 2 ;;
    --output) [[ $# -ge 2 ]] || usage; output=$2; shift 2 ;;
    --device) [[ $# -ge 2 ]] || usage; device=$2; shift 2 ;;
    --dtype) [[ $# -ge 2 ]] || usage; dtype=$2; shift 2 ;;
    --max-tokens) [[ $# -ge 2 ]] || usage; max_tokens=$2; shift 2 ;;
    --prompt) [[ $# -ge 2 ]] || usage; prompt=$2; shift 2 ;;
    --no-thinking) no_thinking=1; shift ;;
    --reasoning-effort) [[ $# -ge 2 ]] || usage; reasoning_effort=$2; shift 2 ;;
    --allow-compatibility-shims) allow_compatibility_shims=1; shift ;;
    -h|--help) usage 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage ;;
  esac
done

[[ -n "$model_dir" ]] || { printf '%s\n' '--model is required' >&2; usage; }
python_bin="$venv_dir/bin/python"
if ((setup)); then
  if [[ ! -x "$python_bin" ]]; then
    bootstrap_python=${SPARK_DEAD_RECKONING_BOOTSTRAP_PYTHON:-python3}
    command -v "$bootstrap_python" >/dev/null 2>&1 || {
      printf 'Bootstrap Python not found: %s\n' "$bootstrap_python" >&2
      exit 1
    }
    "$bootstrap_python" -m venv "$venv_dir"
  fi
  "$python_bin" -m pip install -r "$repo_root/scripts/spark-dead-reckoning-requirements.txt"
fi
[[ -x "$python_bin" ]] || { printf 'Python executable not found: %s\n' "$python_bin" >&2; exit 1; }
[[ -d "$model_dir" ]] || { printf 'Model directory not found: %s\n' "$model_dir" >&2; exit 1; }

runner="$repo_root/core/src/test/resources/dead-reckoning/dead_reckoning_spark_trace.py"
args=("$runner" --model "$model_dir" --output "$output" --device "$device" --dtype "$dtype" --max-tokens "$max_tokens" --reasoning-effort "$reasoning_effort")
if [[ -n "$prompt" ]]; then args+=(--prompt "$prompt"); fi
if ((no_thinking)); then args+=(--no-thinking); fi
if ((allow_compatibility_shims)); then args+=(--allow-compatibility-shims); fi

exec "$python_bin" "${args[@]}"
