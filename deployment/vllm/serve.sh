#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run deployment/vllm/serve.sh inside WSL (or another Linux environment)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PID_FILE="${PROJECT_ROOT}/.cache/vllm-wsl.pid"
ADAPTER_DIR="${PROJECT_ROOT}/outputs/models/news-finetune"
ENV_FILE="${PROJECT_ROOT}/src/.env"
# Linux filesystem: installing CUDA wheels onto /mnt/c is too slow and fragile.
VENV_DIR="${HOME}/.cache/news-finetuning/venv"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export HF_HOME="${PROJECT_ROOT}/.hf-cache"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(tr -d '[:space:]' < "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "vLLM is already running as PID ${old_pid}. Stop it with deployment/vllm/stop.sh." >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Fine-tuned adapter directory not found: ${ADAPTER_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "Fine-tuned adapter config not found: ${ADAPTER_DIR}/adapter_config.json" >&2
  exit 1
fi
if [[ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" && ! -f "${ADAPTER_DIR}/adapter_model.bin" ]]; then
  echo "Fine-tuned adapter weights not found in ${ADAPTER_DIR}" >&2
  exit 1
fi

if [[ ! -d "${HF_HOME}" ]]; then
  mkdir -p "${HF_HOME}"
fi

if [[ -f "${ENV_FILE}" ]]; then
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    export "${line}"
  done < "${ENV_FILE}"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU is not visible in this Linux environment. Install the NVIDIA Windows driver with WSL support, then rerun nvidia-smi inside WSL." >&2
  exit 1
fi

if [[ -x "${HOME}/.local/bin/uv" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
elif [[ -x "${HOME}/.cargo/bin/uv" ]]; then
  export PATH="${HOME}/.cargo/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed in WSL. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/.cache" "${VENV_DIR%/*}"
uv python install 3.12
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv --python 3.12 "${VENV_DIR}"
fi
# uv.lock resolves xgrammar 0.2.4, which has no cp312 manylinux x86_64 wheel.
# vLLM 0.7.2 accepts xgrammar>=0.1.6; 0.1.11 has the needed Linux wheel.
# transformers 5.x removed tokenizer attributes that vLLM 0.7.2 still reads.
uv pip install --python "${VENV_DIR}/bin/python" \
  "vllm==0.7.2" \
  "xgrammar==0.1.11" \
  "transformers==4.48.3"

cleanup() {
  rm -f "${PID_FILE}"
}
trap cleanup EXIT INT TERM

"${VENV_DIR}/bin/python" -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype half \
  --gpu-memory-utilization 0.90 \
  --swap-space 2 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules "news-lora=${ADAPTER_DIR}" \
  --middleware src.serving.middleware.enforce_chinese_suppression \
  --logits-processor-pattern '^src\.models\.vllm_logits_processors\.ChineseTokenSuppressor$' \
  &
server_pid=$!
echo "${server_pid}" > "${PID_FILE}"
echo "vLLM PID ${server_pid} listening on 0.0.0.0:8000 (news-lora)."
wait "${server_pid}"
