#!/usr/bin/env bash
# Fill deployment environment first. This package does not include server models.
set -euo pipefail
if [[ ${1:-} == --help || $# -ne 1 ]]; then
  echo 'Usage: bash server_launch_examples.sh {fastwam|alpha|joint|tri|coarse_fine|pi05}'
  echo 'Required: SERVER_CODE_ROOT, RUNTIME_ENV, INFERENCE_RUN_DIR, INFERENCE_OUTPUT'
  echo 'Optional: EXTRA_PYTHONPATH, SERVER_HOST, SERVER_PORT; FastWAM also requires DATA_ROOT'
  exit 0
fi
family=$1
case "$family" in fastwam|alpha|joint|tri|coarse_fine|pi05) ;; *) echo "Unknown model: $family" >&2; exit 2;; esac
: "${SERVER_CODE_ROOT:?Set the full model/server repository path}"
: "${RUNTIME_ENV:?Set the matched runtime environment script}"
: "${INFERENCE_RUN_DIR:?Set the selected training run directory}"
: "${INFERENCE_OUTPUT:?Set a fresh runtime output directory}"
source "$RUNTIME_ENV"
: "${RUNTIME_PYTHON:?Runtime environment must define RUNTIME_PYTHON}"
export PYTHONPATH="$SERVER_CODE_ROOT${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
cd "$SERVER_CODE_ROOT"
checkpoint=${INFERENCE_CHECKPOINT:-$INFERENCE_RUN_DIR/checkpoint_step_30000.safetensors}
case "$family" in
  alpha)
    extra=()
    if [[ -n ${ACCEPTANCE_SAMPLES:-} ]]; then extra=(--acceptance-samples "$ACCEPTANCE_SAMPLES"); fi
    exec "$RUNTIME_PYTHON" -u scripts/mixed76_server/alpha/serve_tcp_v4.py \
      --checkpoint-dir "$INFERENCE_RUN_DIR" --output "$INFERENCE_OUTPUT" \
      --host "${SERVER_HOST:-127.0.0.1}" --port "${SERVER_PORT:-8866}" "${extra[@]}"
    ;;
  fastwam)
    : "${DATA_ROOT:?Set the source dataset metadata root}"
    # Existing adapter binds loopback, not SERVER_HOST.
    exec "$RUNTIME_PYTHON" -u scripts/mixed76_server/fastwam/serve_v4.py \
      --run "$INFERENCE_RUN_DIR" --weights "$checkpoint" --data-root "$DATA_ROOT" \
      --output "$INFERENCE_OUTPUT" --port "${SERVER_PORT:-18849}"
    ;;
  pi05)
    export JAX_PLATFORMS=cpu
    extra=()
    if [[ -n ${ACCEPTANCE_SAMPLES:-} ]]; then extra=(--acceptance-samples "$ACCEPTANCE_SAMPLES"); fi
    exec "$RUNTIME_PYTHON" -u scripts/mixed76_server/pi05/serve_pi05_v4.py \
      --checkpoint-dir "$INFERENCE_RUN_DIR" --output "$INFERENCE_OUTPUT" \
      --host "${SERVER_HOST:-127.0.0.1}" --port "${SERVER_PORT:-8858}" --steps 10 "${extra[@]}"
    ;;
  joint|tri|coarse_fine)
    exec "$RUNTIME_PYTHON" -u -m openwam.deploy.ours.tcp_v4 \
      --model "$family" --checkpoint "$checkpoint" --output "$INFERENCE_OUTPUT" \
      --host "${SERVER_HOST:-127.0.0.1}" --port "${SERVER_PORT:-8861}"
    ;;
esac
