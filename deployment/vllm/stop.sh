#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run deployment/vllm/stop.sh inside WSL (or another Linux environment)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PID_FILE="${PROJECT_ROOT}/.cache/vllm-wsl.pid"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(tr -d '[:space:]' < "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    kill "${old_pid}"
    echo "Stopped vLLM PID ${old_pid}."
  else
    echo "No running vLLM process for PID file ${PID_FILE}."
  fi
  rm -f "${PID_FILE}"
  exit 0
fi

echo "vLLM PID file not found; nothing to stop."
