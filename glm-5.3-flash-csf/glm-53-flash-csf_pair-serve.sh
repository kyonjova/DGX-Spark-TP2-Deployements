#!/usr/bin/env bash
# glm-53-flash-csf_pair-serve.sh
#
# Validate or start one rank of a two-node DGX Spark pair serving
# local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD (NVFP4 QAD routed
# experts with losslessly compressed scales, MXFP8 attention and shared
# experts, NVFP4 MTP experts) at TP=2 under the karmic-kraken INTEGRATION
# image (build-kk-integration-cache-cu132.env: vLLM f1c2508f + b12x 52640cb1,
# CuTe DSL 4.7.1).
#
# What is different from the glm-5.3-flash-cache launcher:
#   * FP4-CSF serving directory. The checkpoint keeps the Hugging Face files
#     under metadata/ and the compressed tensors under tensors/. vLLM cannot
#     open that layout directly: it needs a directory whose config.json names
#     the CSF reader (quant_method nvfp4_csf, format_version 1, absolute
#     checkpoint_root, the original config under source_quantization_config).
#     This launcher builds that directory on the host (python3), exactly as
#     blackwell-llm-docker runtime/launcher.py prepare_csf_checkpoint() does,
#     mounts it read-only, and points checkpoint_root at the weights mount.
#     --quantization nvfp4_csf --load-format nvfp4_csf are fixed: the CSF
#     reader owns the routed experts, so no other loader applies.
#   * Expert precision as three named choices (EXPERT_ACTIVATIONS,
#     ROUTER_WEIGHTS, PREFILL_ACTIVATIONS) mapped to the b12x/vLLM variables
#     the same way the release launcher maps them. The env file must not set
#     the raw variables; the launcher passes them with -e.
#   * Optional decode context parallelism (DCP_SIZE=2) with the derived
#     interleave and CKV-gather settings.
#   * FABRIC_PROFILE (single | dualpath | dual) with per-device RoCEv2 GID
#     checks, as in the Qwen launcher.
#   * CUDAGRAPH_CAPTURE_SIZES / MAX_CUDAGRAPH_CAPTURE_SIZE auto: capture
#     exactly the decode shapes MTP/DFlash produce.
#   * No external prefix cache (SparkCache / LMCache): GPU prefix caching only.
#
# Usage:
#     ./glm-53-flash-csf_pair-serve.sh --check   rank-0.env
#     ./glm-53-flash-csf_pair-serve.sh --run     rank-0.env
#     ./glm-53-flash-csf_pair-serve.sh --restart rank-0.env
#     ./glm-53-flash-csf_pair-serve.sh --logs    rank-0.env
#     ./glm-53-flash-csf_pair-serve.sh --verify  rank-0.env
#     ./glm-53-flash-csf_pair-serve.sh --status  rank-0.env
#     ./glm-53-flash-csf_pair-serve.sh --down    rank-0.env
#     ./glm-53-flash-csf_pair-serve.sh --clear   rank-0.env
#     ./glm-53-flash-csf_pair-serve.sh --fresh   rank-0.env
#
# Anything after ENV_FILE is appended verbatim to the vllm serve argv.
# Start rank 1 (headless, waits for rank 0) before rank 0.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: glm-53-flash-csf_pair-serve.sh [MODE] ENV_FILE [extra vllm args...]

  --check     validate the env file, build the CSF serving dir, print the launch command (default)
  --run       validate, then start the container detached
  --restart   stop and remove this rank's container, then run
  --down      stop and remove this rank's container
  --logs      follow this rank's container logs
  --verify    grep this rank's log for the expected startup markers (+ /health on rank 0)
  --status    show this rank's container state
  --clear     wipe CACHE_HOST_PATH contents (container must not exist)
  --fresh     drop page cache (sync + sudo), then --run, then follow logs
EOF
}

die() {
  echo "glm53-csf pair launcher: $*" >&2
  exit 20
}

warn() {
  echo "glm53-csf pair launcher: warning: $*" >&2
}

# Reclaim page cache. On this vLLM pin the startup gate reads MemAvailable on
# GB10 (integrated GPU -> psutil), so page cache no longer blocks the gate;
# --fresh still matters for a clean, comparable benchmark boot (the CSF reader
# mmaps the shards and leaves ~100 GB of page cache behind per node).
drop_page_cache() {
  [ -e /proc/sys/vm/drop_caches ] \
    || die "cannot drop page cache: /proc/sys/vm/drop_caches does not exist (not Linux?)"
  printf 'sync + drop_caches=3 (reclaiming page cache)...\n'
  sync
  if [ "$(id -u)" = 0 ]; then
    echo 3 > /proc/sys/vm/drop_caches
  elif command -v sudo >/dev/null 2>&1; then
    printf '3' | sudo tee /proc/sys/vm/drop_caches >/dev/null
  else
    die "dropping page cache needs root; run as root or install sudo"
  fi
  awk '/^MemFree:/{printf "  MemFree after drop: %.1f GiB\n", $2/1048576}' /proc/meminfo
}

# ---------------------------------------------------------------- arguments

mode=--check
fresh_follow=0
case "${1:-}" in
  --check|--run|--restart|--down|--logs|--status|--clear|--verify|--fresh) mode=$1; shift ;;
  -h|--help)     usage; exit 0 ;;
  --*)           usage; exit 64 ;;
esac

env_file=${1:-}
[ -n "$env_file" ] || { usage; exit 64; }
shift
passthrough=("$@")

[ -f "$env_file" ] || die "environment file is missing: $env_file"
env_file=$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [ "$mode" = --fresh ]; then
  drop_page_cache
  mode=--run
  fresh_follow=1
fi

# CRLF breaks both the shell source below and docker --env-file.
if grep -qU $'\r' "$env_file" 2>/dev/null; then
  die "environment file has CRLF line endings; convert with: sed -i 's/\r$//' $env_file"
fi

if grep -Ev '^[[:space:]]*(#|$)' "$env_file" \
   | grep -Eq '<[A-Za-z0-9_]+>|REPLACE_WITH_'; then
  die "environment file contains unresolved placeholders: $env_file"
fi

# The file is sourced here AND passed to docker --env-file: a space ends the
# shell assignment, and docker keeps trailing comments as part of the value.
if grep -nE '^[A-Z][A-Z0-9_]*=[^#]*[[:space:]]' "$env_file" | grep -vE '^[0-9]+:[A-Z][A-Z0-9_]*=[[:space:]]*$' | head -3 | grep -q .; then
  grep -nE '^[A-Z][A-Z0-9_]*=[^#]*[[:space:]]' "$env_file" | grep -vE '^[0-9]+:[A-Z][A-Z0-9_]*=[[:space:]]*$' | head -3 >&2
  die "environment file has values containing whitespace or trailing comments (lines above); quote nothing, put comments on their own line"
fi

# The expert-precision variables have ONE owner: the launcher derives them
# from EXPERT_ACTIVATIONS / ROUTER_WEIGHTS / PREFILL_ACTIVATIONS and passes
# them with -e. A raw copy in the env file would reach the container through
# --env-file and make the file lie about what ran.
for raw in VLLM_B12X_MOE_FP4_FORCE_A16 B12X_W4A16_FP32_TOPK_WEIGHTS \
           B12X_W4A16_A4_PREFILL_MIN_TOKENS VLLM_B12X_NVFP4_ACTIVATION_MODE \
           VLLM_B12X_MLA_CKV_GATHER; do
  if grep -Eq "^[[:space:]]*${raw}=" "$env_file"; then
    die "$raw is derived by the launcher; remove it from $env_file (set EXPERT_ACTIVATIONS / ROUTER_WEIGHTS / PREFILL_ACTIVATIONS / DCP_SIZE instead)"
  fi
done

# shellcheck disable=SC1090
. "$env_file"

# ------------------------------------------------- container management modes

: "${CONTAINER_RUNTIME:=docker}"
: "${CONTAINER_NAME_SUFFIX:=}"
# Same name as the other GLM-5.3 launchers on purpose: only one GLM server can
# own a node's memory, so a second launcher refuses while the first is up.
mgmt_container="glm53-flash-r${NODE_RANK:-?}${CONTAINER_NAME_SUFFIX}"

require_runtime() {
  command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1 \
    || die "$CONTAINER_RUNTIME is unavailable"
}

container_down() {
  require_runtime
  if "$CONTAINER_RUNTIME" container inspect "$mgmt_container" >/dev/null 2>&1; then
    printf 'stopping %s\n' "$mgmt_container"
    "$CONTAINER_RUNTIME" stop -t "${STOP_TIMEOUT:-30}" "$mgmt_container" >/dev/null
    "$CONTAINER_RUNTIME" rm "$mgmt_container" >/dev/null
    printf 'removed %s\n' "$mgmt_container"
  else
    printf 'no such container: %s\n' "$mgmt_container"
  fi
  # A leftover LMCache sidecar from the glm-5.3-flash-cache launcher would
  # hold /dev/shm and host RAM; remove it too.
  if "$CONTAINER_RUNTIME" container inspect "${mgmt_container}-lmcache" >/dev/null 2>&1; then
    printf 'stopping leftover %s\n' "${mgmt_container}-lmcache"
    "$CONTAINER_RUNTIME" stop -t 10 "${mgmt_container}-lmcache" >/dev/null
    "$CONTAINER_RUNTIME" rm "${mgmt_container}-lmcache" >/dev/null
  fi
}

container_clear() {
  local target=${CACHE_HOST_PATH-} count
  case "$target" in
    /*) ;;
    *) die "CACHE_HOST_PATH must be an absolute host path: ${target:-<unset>}" ;;
  esac
  [ -d "$target" ] || die "CACHE_HOST_PATH is not a directory: $target"
  [ -w "$target" ] || die "CACHE_HOST_PATH is not writable: $target"
  case "$target" in
    /|/root|/home|/usr|/var|/etc|/opt|/tmp|/mnt|/models|/data)
      die "refusing to clear a system path: $target" ;;
  esac
  [ "$(printf '%s' "$target" | tr -cd / | wc -c)" -ge 4 ] \
    || die "refusing to clear a shallow path (needs 4+ levels): $target"
  [ "$target" != "${MODEL_HOST_PATH-}" ] \
    || die "CACHE_HOST_PATH equals MODEL_HOST_PATH; refusing to clear: $target"
  [ "$target" != "${DFLASH_MODEL_HOST_PATH-}" ] \
    || die "CACHE_HOST_PATH equals DFLASH_MODEL_HOST_PATH; refusing to clear: $target"
  case "${MODEL_HOST_PATH-}/" in
    "$target"/*) die "MODEL_HOST_PATH lives inside CACHE_HOST_PATH; refusing to clear: $target" ;;
  esac

  require_runtime
  if "$CONTAINER_RUNTIME" container inspect "$mgmt_container" >/dev/null 2>&1; then
    die "container $mgmt_container still exists; run --down first"
  fi

  count=$(find "$target" -mindepth 1 -maxdepth 1 | wc -l)
  if [ "$count" -eq 0 ]; then
    printf 'already empty: %s\n' "$target"
    return 0
  fi
  printf 'about to delete %s entries under %s (%s)\n' \
    "$count" "$target" "$(du -sh "$target" 2>/dev/null | cut -f1)"
  { find "$target" -mindepth 1 -maxdepth 1 -printf '  %f\n' 2>/dev/null || true; } | head -20 || true
  if [ "${CLEAR_ASSUME_YES:-0}" != 1 ]; then
    printf 'type "clear" to confirm: '
    read -r reply
    [ "$reply" = clear ] || die "aborted"
  fi
  find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  printf 'cleared %s\n' "$target"
  printf 'note: next boot recompiles every B12X/CuTe DSL, Triton and FlashInfer artifact (long first boot)\n'
}

case "$mode" in
  --clear)  container_clear; exit 0 ;;
  --down)   container_down;  exit 0 ;;
  --logs)
    require_runtime
    exec "$CONTAINER_RUNTIME" logs -f "$mgmt_container"
    ;;
  --verify)
    require_runtime
    # Markers that prove the intended path was taken. Absence of one you
    # expect means a fallback: e.g. no CSF line = not the CSF reader, no A4
    # line with PREFILL_ACTIVATIONS=a4 = A4 prefill not compiled.
    "$CONTAINER_RUNTIME" logs "$mgmt_container" 2>&1 | grep -E \
      'speculative_config|nvfp4_csf|NVFP4-CSF|CSF|A4 prefill|prefill with A4|W4A16|Using .* all-reduce backends|RoCEnante|B12X_ROCENANTE|kda_prefill|KDA prefill|FlashAttention version 2|split GLM-5.3 cache pages|physical page sizes|attention block size|decode_context_parallel|Available KV cache memory|GPU KV cache size|Maximum concurrency|Model loading took|Graph capturing finished|cudagraph_mode=|fastokens|Application startup complete' \
      | sed 's/^/  /'
    if [ "${NODE_RANK:-}" = 0 ]; then
      printf 'health: '; curl -fsS "http://127.0.0.1:${API_PORT:-8000}/health" 2>/dev/null && echo " OK" || echo " not ready"
    fi
    exit 0
    ;;
  --status)
    require_runtime
    "$CONTAINER_RUNTIME" ps -a --filter "name=^/${mgmt_container}$" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'
    exit 0
    ;;
  --restart)
    container_down
    mode=--run
    ;;
esac

# --------------------------------------------------------------- validators

require_value() {
  local name=$1
  [ -n "${!name-}" ] || die "required value is empty: $name"
}

require_directory() {
  local name=$1 value
  value=${!name-}
  case "$value" in
    /*) ;;
    *) die "$name must be an absolute host path: $value" ;;
  esac
  [ -d "$value" ] || die "$name directory does not exist: $value"
}

require_positive_integer() {
  local name=$1 value
  value=${!name-}
  case "$value" in
    ''|*[!0-9]*) die "$name must be a positive integer: $value" ;;
  esac
  [ "$((10#$value))" -gt 0 ] || die "$name must be greater than zero"
}

require_nonnegative_integer() {
  local name=$1 value
  value=${!name-}
  case "$value" in
    ''|*[!0-9]*) die "$name must be a non-negative integer: $value" ;;
  esac
}

require_port() {
  local name=$1
  require_positive_integer "$name"
  [ "$((10#${!name}))" -le 65535 ] || die "$name must be in 1..65535: ${!name}"
}

require_bool() {
  local name=$1
  case "${!name-}" in
    0|1) ;;
    *) die "$name must be 0 or 1: ${!name-}" ;;
  esac
}

require_unit_fraction() {
  local name=$1 value
  value=${!name-}
  awk -v v="$value" 'BEGIN{ exit !(v+0 > 0 && v+0 <= 1 && v ~ /^[0-9]*\.?[0-9]+$/) }' \
    || die "$name must be a fraction in (0,1]: $value"
}

# ------------------------------------------------------- serving defaults
# Every value below may be set in the env file; these are the fallbacks and
# follow the release's glm53-tp2 preset where it is not PCIe-specific.

: "${SERVED_MODEL_NAME:=GLM-5.3-Flash}"
: "${SPECULATOR:=mtp}"                 # mtp | dflash2 | none
: "${DCP_SIZE:=1}"                     # 1 | 2 (decode context parallelism across the pair)
: "${EXPERT_ACTIVATIONS:=bf16}"        # bf16 (W4A16, QAD-qualified) | fp4 (W4A4)
: "${ROUTER_WEIGHTS:=fp32}"            # fp32 | bf16 (fp32 needs EXPERT_ACTIVATIONS=bf16)
: "${PREFILL_ACTIVATIONS:=a4}"         # a4 (NVFP4 activations for prefill rows) | a16
: "${A4_PREFILL_MIN_TOKENS:=1536}"     # release value (GLM_A4_PREFILL_MIN_TOKENS)
: "${GPU_MEMORY_UTILIZATION:=0.85}"
: "${KV_CACHE_MEMORY_BYTES:=}"
: "${KV_CACHE_DTYPE:=fp8}"
: "${DFLASH_KV_CACHE_DTYPE:=auto}"
: "${DFLASH_ATTENTION_BACKEND:=FLASH_ATTN}"
: "${PREFILL_SCHEDULE_INTERVAL:=1}"
: "${PREFILL_COMPUTE_SHARE:=0.4}"
: "${PREFILL_COMPUTE_HALF_LIFE:=}"
: "${MAX_PARALLEL_PREFILLS:=1}"
: "${FAIRNESS_ENGINE:=}"
: "${COMPILATION_LEVEL:=}"
: "${GENERATION_CONFIG:=auto}"
: "${SAMPLING_TEMPERATURE:=}"
: "${SAMPLING_TOP_P:=}"
: "${SAMPLING_TOP_K:=}"
: "${SAMPLING_MIN_P:=}"
: "${SAMPLING_REPETITION_PENALTY:=}"
: "${REASONING_EFFORT:=high}"
: "${CLEAR_THINKING:=0}"
: "${KDA_PREFILL_BACKEND:=b12x}"
: "${KDA_DECODE_BACKEND:=}"
: "${GDN_DECODE_KERNEL:=}"
: "${RECURRENT_CHECKPOINT_POLICY:=request_boundaries}"
: "${REPLAYSSM:=}"
: "${DRAFT_SAMPLE_METHOD:=probabilistic}"
: "${REJECTION_SAMPLE_METHOD:=standard}"
: "${MTP_MOE_BACKEND:=b12x}"           # NVFP4 MTP experts in this checkpoint -> b12x
: "${MTP_ATTENTION_BACKEND:=B12X}"
: "${BLOCK_SIZE:=256}"
: "${LANGUAGE_MODEL_ONLY:=0}"
: "${LIMIT_MM:=1}"
: "${MM_IMAGES:=32}"
: "${MM_VIDEOS:=0}"
: "${MM_PROCESSOR_CACHE_GB:=0}"
: "${MM_ENCODER_TP_MODE:=data}"
: "${CHAT_TEMPLATE_HOST_PATH=}"        # "=" not ":=": empty means the checkpoint's own template
: "${TORCH_PROFILE_HOST_DIR:=}"
: "${CONTAINER_MEMORY_GB:=}"
: "${ADAPTIVE_SPECULATIVE_TOKENS:=0}"
: "${ADAPTIVE_SPECULATIVE_TOKENS_WINDOW:=32}"
: "${API_KEY:=}"
: "${ENABLE_FLASHINFER_AUTOTUNE:=0}"
: "${MOE_BACKEND:=b12x}"
: "${ATTENTION_BACKEND:=B12X}"
: "${LINEAR_BACKEND:=b12x}"
: "${MAX_MODEL_LEN:=1048576}"
: "${MAX_NUM_SEQS:=8}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"
: "${MAX_CUDAGRAPH_CAPTURE_SIZE:=auto}"
: "${CUDAGRAPH_CAPTURE_SIZES=auto}"    # "=" not ":=": an empty line means engine default grid
: "${CUDAGRAPH_MODE:=FULL_AND_PIECEWISE}"
: "${ENABLE_PREFIX_CACHING:=1}"
: "${ENABLE_CHUNKED_PREFILL:=1}"
: "${SERVING_IMAGE:=local/vllm:karmic-kraken-integration-cache-cu132}"
: "${SHM_SIZE:=16g}"
: "${FABRIC_PROFILE:=single}"
: "${B12X_ROCE_HCA:=}"
: "${VLLM_USE_FASTOKENS:=0}"
: "${CHECKPOINT_MANIFEST_SHA256:=}"
: "${MEM_PREFLIGHT:=die}"

case "$SPECULATOR" in
  mtp)     : "${NUM_SPECULATIVE_TOKENS:=3}" ;;
  dflash2) : "${NUM_SPECULATIVE_TOKENS:=7}" ;;
  none)    : "${NUM_SPECULATIVE_TOKENS:=0}" ;;
  *) die "SPECULATOR must be mtp, dflash2, or none: $SPECULATOR" ;;
esac
case "$NUM_SPECULATIVE_TOKENS" in
  ''|*[!0-9]*) die "NUM_SPECULATIVE_TOKENS must be a non-negative integer: $NUM_SPECULATIVE_TOKENS" ;;
esac
[ "$SPECULATOR" != none ] || [ "$NUM_SPECULATIVE_TOKENS" = 0 ] \
  || die "SPECULATOR=none with NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS"
[ "$SPECULATOR" = none ] || [ "$NUM_SPECULATIVE_TOKENS" -gt 0 ] \
  || die "SPECULATOR=$SPECULATOR needs NUM_SPECULATIVE_TOKENS > 0 (or SPECULATOR=none)"
# Upstream's GLM KDA recovery (REPLAYSSM auto) covers 1..7 draft tokens.
[ "$NUM_SPECULATIVE_TOKENS" -le 7 ] \
  || warn "NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS exceeds the 7 tokens GLM KDA recovery supports; the engine falls back to full speculative states"

case "$ADAPTIVE_SPECULATIVE_TOKENS" in 0|1) : ;; *) die "ADAPTIVE_SPECULATIVE_TOKENS must be 0 or 1" ;; esac
if [ "$ADAPTIVE_SPECULATIVE_TOKENS" = 1 ]; then
  [ "$SPECULATOR" = mtp ] || die "ADAPTIVE_SPECULATIVE_TOKENS requires SPECULATOR=mtp"
  : "${ADAPTIVE_SPECULATIVE_TOKENS_INITIAL:=$(( NUM_SPECULATIVE_TOKENS < 3 ? NUM_SPECULATIVE_TOKENS : 3 ))}"
  require_positive_integer ADAPTIVE_SPECULATIVE_TOKENS_INITIAL
  require_positive_integer ADAPTIVE_SPECULATIVE_TOKENS_WINDOW
  [ "$ADAPTIVE_SPECULATIVE_TOKENS_INITIAL" -le "$NUM_SPECULATIVE_TOKENS" ] \
    || die "ADAPTIVE_SPECULATIVE_TOKENS_INITIAL must not exceed NUM_SPECULATIVE_TOKENS"
fi

require_positive_integer PREFILL_SCHEDULE_INTERVAL
case "$COMPILATION_LEVEL" in ""|0|1|2|3) : ;; *) die "COMPILATION_LEVEL must be empty or 0-3: $COMPILATION_LEVEL" ;; esac
case "$GENERATION_CONFIG" in auto|vllm) : ;; *) die "GENERATION_CONFIG must be auto or vllm: $GENERATION_CONFIG" ;; esac
case "$KDA_PREFILL_BACKEND" in ""|auto|triton|flashkda|flashinfer|b12x) : ;; *) die "KDA_PREFILL_BACKEND must be empty, auto, triton, flashkda, flashinfer, or b12x: $KDA_PREFILL_BACKEND" ;; esac
case "$KDA_DECODE_BACKEND" in ""|auto|native|flashinfer|triton) : ;; *) die "KDA_DECODE_BACKEND must be empty, auto, native, flashinfer, or triton: $KDA_DECODE_BACKEND" ;; esac
case "$GDN_DECODE_KERNEL" in ""|b12x|cuda|triton) : ;; *) die "GDN_DECODE_KERNEL must be empty, b12x, cuda, or triton: $GDN_DECODE_KERNEL" ;; esac
case "$REPLAYSSM" in ""|0|1) : ;; *) die "REPLAYSSM must be empty, 0, or 1: $REPLAYSSM" ;; esac
[ -z "$FAIRNESS_ENGINE" ] || die "FAIRNESS_ENGINE=$FAIRNESS_ENGINE: --fairness-engine does not exist on karmic-kraken; set PREFILL_COMPUTE_SHARE alone (with PREFILL_SCHEDULE_INTERVAL=1)"
case "$DRAFT_SAMPLE_METHOD" in greedy|probabilistic) : ;; *) die "DRAFT_SAMPLE_METHOD must be greedy or probabilistic: $DRAFT_SAMPLE_METHOD" ;; esac
case "$REJECTION_SAMPLE_METHOD" in standard|block|synthetic) : ;; *) die "REJECTION_SAMPLE_METHOD must be standard, block, or synthetic: $REJECTION_SAMPLE_METHOD" ;; esac
case "$MTP_MOE_BACKEND" in
  b12x|auto) : ;;
  marlin|humming) warn "MTP_MOE_BACKEND=$MTP_MOE_BACKEND: this checkpoint's MTP experts are NVFP4 (W4A16 group), which b12x serves -- the release preset runs b12x. marlin/humming were the workaround for the ORIGINAL checkpoint's MXFP8 MTP experts" ;;
  *) die "MTP_MOE_BACKEND must be b12x, auto, marlin, or humming: $MTP_MOE_BACKEND" ;;
esac
case "$MTP_ATTENTION_BACKEND" in B12X|auto) : ;; *) die "MTP_ATTENTION_BACKEND must be B12X or auto: $MTP_ATTENTION_BACKEND" ;; esac
case "$DFLASH_ATTENTION_BACKEND" in ""|FLASH_ATTN|B12X|auto) : ;; *) die "DFLASH_ATTENTION_BACKEND must be empty, FLASH_ATTN, B12X, or auto: $DFLASH_ATTENTION_BACKEND" ;; esac
case "$MAX_PARALLEL_PREFILLS" in ""|auto) : ;; *) require_positive_integer MAX_PARALLEL_PREFILLS ;; esac
case "$RECURRENT_CHECKPOINT_POLICY" in ""|auto|request_boundaries|aligned) : ;; *) die "RECURRENT_CHECKPOINT_POLICY must be empty, auto, request_boundaries, or aligned: $RECURRENT_CHECKPOINT_POLICY" ;; esac
for name in SAMPLING_TEMPERATURE SAMPLING_TOP_P SAMPLING_TOP_K SAMPLING_MIN_P SAMPLING_REPETITION_PENALTY; do
  v=${!name-}
  [ -z "$v" ] || awk -v v="$v" 'BEGIN{ exit !(v ~ /^[0-9]*\.?[0-9]+$/) }' || die "$name must be a number: $v"
done
case "$REASONING_EFFORT" in ""|low|high|max) : ;; *) die "REASONING_EFFORT must be empty, low, high, or max: $REASONING_EFFORT" ;; esac
case "$CLEAR_THINKING" in ""|0|1) : ;; *) die "CLEAR_THINKING must be empty, 0, or 1: $CLEAR_THINKING" ;; esac
if [ -n "$PREFILL_COMPUTE_SHARE" ]; then
  # SchedulerConfig: a fraction in (0,1) or auto; cannot be combined with an
  # interval greater than one. The half-life is auto-mode only.
  [ "$PREFILL_COMPUTE_SHARE" = auto ] \
    || awk -v v="$PREFILL_COMPUTE_SHARE" 'BEGIN{ exit !(v+0 > 0 && v+0 < 1 && v ~ /^[0-9]*\.?[0-9]+$/) }' \
    || die "PREFILL_COMPUTE_SHARE must be auto or a fraction in (0,1): $PREFILL_COMPUTE_SHARE"
  [ "$PREFILL_SCHEDULE_INTERVAL" = 1 ] \
    || die "PREFILL_COMPUTE_SHARE=$PREFILL_COMPUTE_SHARE requires PREFILL_SCHEDULE_INTERVAL=1 (the engine rejects share + interval > 1)"
  if [ -n "$PREFILL_COMPUTE_HALF_LIFE" ]; then
    [ "$PREFILL_COMPUTE_SHARE" = auto ] || die "PREFILL_COMPUTE_HALF_LIFE is only valid with PREFILL_COMPUTE_SHARE=auto"
  fi
elif [ -n "$PREFILL_COMPUTE_HALF_LIFE" ]; then
  die "PREFILL_COMPUTE_HALF_LIFE requires PREFILL_COMPUTE_SHARE=auto"
fi
require_positive_integer BLOCK_SIZE
case "$MM_PROCESSOR_CACHE_GB" in ""|[0-9]*) : ;; *) die "MM_PROCESSOR_CACHE_GB must be a number: $MM_PROCESSOR_CACHE_GB" ;; esac
case "$MM_ENCODER_TP_MODE" in ""|data|weights) : ;; *) die "MM_ENCODER_TP_MODE must be empty, data, or weights: $MM_ENCODER_TP_MODE" ;; esac
require_bool VLLM_USE_FASTOKENS

# ---------------------------------------------------- expert precision
# Same mapping as blackwell-llm-docker runtime/launcher.py (GLM precision
# options):
#   EXPERT_ACTIVATIONS bf16 -> VLLM_B12X_MOE_FP4_FORCE_A16=1 (W4A16: FP4
#     weights, BF16 activations); fp4 -> 0 (W4A4).
#   ROUTER_WEIGHTS fp32 -> B12X_W4A16_FP32_TOPK_WEIGHTS=1: W4A16 expert
#     outputs combined with FP32 router weights (bf16 experts only).
#   PREFILL_ACTIVATIONS a4 -> B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536: prefill
#     rows of W4A16 MoE calls run with NVFP4 activations over the same packed
#     weights; decode rows always stay W4A16. a16 -> 0.
# The checkpoint card asks for FORCE_A16=1 + FP32 top-k weights; the release
# measured a4 "as accurate as a16" on this QAD checkpoint and 12-14% faster
# prefill at TP4 (and refuses a4 only for the pre-QAD -Spark checkpoint).
case "$EXPERT_ACTIVATIONS" in bf16|fp4) : ;; *) die "EXPERT_ACTIVATIONS must be bf16 or fp4: $EXPERT_ACTIVATIONS" ;; esac
case "$ROUTER_WEIGHTS" in fp32|bf16) : ;; *) die "ROUTER_WEIGHTS must be fp32 or bf16: $ROUTER_WEIGHTS" ;; esac
case "$PREFILL_ACTIVATIONS" in a4|a16) : ;; *) die "PREFILL_ACTIVATIONS must be a4 or a16: $PREFILL_ACTIVATIONS" ;; esac
require_positive_integer A4_PREFILL_MIN_TOKENS
if [ "$EXPERT_ACTIVATIONS" = fp4 ]; then
  [ "$ROUTER_WEIGHTS" = bf16 ] \
    || die "ROUTER_WEIGHTS=fp32 applies to W4A16 experts only; with EXPERT_ACTIVATIONS=fp4 set ROUTER_WEIGHTS=bf16"
  [ "$PREFILL_ACTIVATIONS" = a16 ] \
    || die "PREFILL_ACTIVATIONS=a4 needs EXPERT_ACTIVATIONS=bf16 (with fp4 every row is already W4A4); set PREFILL_ACTIVATIONS=a16"
  warn "EXPERT_ACTIVATIONS=fp4 (W4A4 decode) is off the checkpoint card's accuracy settings (W4A16 + FP32 router weights); quality A/B only"
fi
moe_force_a16=0; [ "$EXPERT_ACTIVATIONS" = bf16 ] && moe_force_a16=1
fp32_topk=0; [ "$ROUTER_WEIGHTS" = fp32 ] && fp32_topk=1
a4_min_tokens=0; [ "$PREFILL_ACTIVATIONS" = a4 ] && a4_min_tokens=$A4_PREFILL_MIN_TOKENS
precision_env_args=(
  -e "VLLM_B12X_MOE_FP4_FORCE_A16=$moe_force_a16"
  -e "B12X_W4A16_FP32_TOPK_WEIGHTS=$fp32_topk"
  -e "B12X_W4A16_A4_PREFILL_MIN_TOKENS=$a4_min_tokens"
)
# Release launcher: GLM without a speculator quantizes the NVFP4 activations
# of the dense path ("GLM non-speculative activation policy").
[ "$SPECULATOR" != none ] || precision_env_args+=(-e VLLM_B12X_NVFP4_ACTIVATION_MODE=quantized)
precision_desc="experts $EXPERT_ACTIVATIONS, router $ROUTER_WEIGHTS, prefill $PREFILL_ACTIVATIONS$([ "$a4_min_tokens" = 0 ] || printf ' (>= %s tokens)' "$a4_min_tokens")"

# ------------------------------------------------- decode context parallel
# DCP=2 stores each request's MLA KV split across the two ranks (twice the
# tokens for the same pin) at the cost of a cross-node gather in every
# decode attention step. The release TP2 preset runs DCP2 on PCIe; Luke's
# GB10 TP2 launcher and our measured profiles run DCP1. A/B, not a default.
case "$DCP_SIZE" in 1|2) : ;; *) die "DCP_SIZE must be 1 or 2 on a TP=2 pair: $DCP_SIZE" ;; esac
dcp_args=(--decode-context-parallel-size "$DCP_SIZE")
dcp_env_args=(-e "VLLM_B12X_MLA_CKV_GATHER=$(( DCP_SIZE > 1 ? 1 : 0 ))")
if [ "$DCP_SIZE" = 2 ]; then
  dcp_args+=(--cp-kv-cache-interleave-size 4 --dcp-kv-cache-interleave-size 4)
  : "${VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS:=65536}"
  dcp_env_args+=(-e "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=$VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS")
fi

# ------------------------------------------------------------ launch checks

for name in \
  NODE_RANK MASTER_ADDR MODEL_HOST_PATH CACHE_HOST_PATH API_PORT MASTER_PORT \
  MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
  LD_PRELOAD VLLM_NCCL_SO_PATH NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME \
  VLLM_HOST_IP NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_IB_HCA \
  NCCL_IB_GID_INDEX NCCL_IB_SUBNET_AWARE_ROUTING NCCL_IB_MERGE_NICS \
  NCCL_PROTO NCCL_P2P_LEVEL NCCL_CROSS_NIC NCCL_CUMEM_ENABLE \
  NCCL_IGNORE_CPU_AFFINITY NCCL_TUNER_PLUGIN CUTE_DSL_ARCH \
  VLLM_ENABLE_PCIE_ALLREDUCE; do
  require_value "$name"
done

case "$NODE_RANK" in
  0|1) ;;
  *) die "NODE_RANK must be 0 or 1: $NODE_RANK" ;;
esac

require_directory MODEL_HOST_PATH
require_directory CACHE_HOST_PATH
[ -r "$MODEL_HOST_PATH" ] || die "MODEL_HOST_PATH is not readable: $MODEL_HOST_PATH"
[ -w "$CACHE_HOST_PATH" ] || die "CACHE_HOST_PATH is not writable: $CACHE_HOST_PATH"
case "$MODEL_HOST_PATH/" in
  "$CACHE_HOST_PATH"/*) die "MODEL_HOST_PATH must not live inside CACHE_HOST_PATH (--clear wipes it)" ;;
esac

if [ "$SPECULATOR" = dflash2 ]; then
  require_value DFLASH_MODEL_HOST_PATH
  require_directory DFLASH_MODEL_HOST_PATH
  [ -f "$DFLASH_MODEL_HOST_PATH/config.json" ] \
    || die "DFLASH_MODEL_HOST_PATH has no config.json: $DFLASH_MODEL_HOST_PATH"
  ls "$DFLASH_MODEL_HOST_PATH"/*.safetensors >/dev/null 2>&1 \
    || die "DFLASH_MODEL_HOST_PATH contains no weight files: $DFLASH_MODEL_HOST_PATH"
  if [ -n "${DFLASH_WEIGHTS_SHA256-}" ]; then
    printf '%s' "$DFLASH_WEIGHTS_SHA256" | grep -Eq '^[0-9a-f]{64}$' \
      || die "DFLASH_WEIGHTS_SHA256 must be 64 lowercase hex chars"
    [ -f "$DFLASH_MODEL_HOST_PATH/model.safetensors" ] \
      || die "DFLASH_WEIGHTS_SHA256 is set but $DFLASH_MODEL_HOST_PATH/model.safetensors does not exist"
    echo "glm53-csf pair launcher: hashing draft weights..." >&2
    draft_sha=$(sha256sum "$DFLASH_MODEL_HOST_PATH/model.safetensors" | cut -d' ' -f1)
    [ "$draft_sha" = "$DFLASH_WEIGHTS_SHA256" ] \
      || die "draft weights sha256 mismatch: got $draft_sha, expected $DFLASH_WEIGHTS_SHA256"
  fi
  [ "$NUM_SPECULATIVE_TOKENS" = 7 ] \
    || warn "DFlash2 is trained for 7 draft tokens (8-token block); NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS is off the trained configuration"
fi

# ------------------------------------------- FP4-CSF checkpoint + serving dir
# Validates the download (manifest schema, every listed shard at its exact
# size, metadata files against their manifest sha256), then writes the
# serving directory vLLM opens. The manifest names every shard and metadata
# file by size/sha256, so its digest identifies the checkpoint content on
# both nodes: compare the printed identity between ranks (or pin it with
# CHECKPOINT_MANIFEST_SHA256 in both env files).
command -v python3 >/dev/null 2>&1 \
  || die "python3 is required on the host to validate the CSF checkpoint and write its serving directory"
model_container_path=/models/glm-5.3-flash-csf
serving_container_path=/models/glm-5.3-flash-csf-serving
csf_serving_parent="$CACHE_HOST_PATH/csf-serving"
mkdir -p "$csf_serving_parent" || die "cannot create $csf_serving_parent"
csf_out=$(python3 - "$MODEL_HOST_PATH" "$model_container_path" "$csf_serving_parent" <<'PYEOF'
import hashlib, json, os, pathlib, re, shutil, sys, tempfile

root = pathlib.Path(sys.argv[1])
container_root = sys.argv[2]
parent = pathlib.Path(sys.argv[3])

def fail(msg):
    print("ERROR " + msg)
    sys.exit(0)

manifest_path = root / "manifest.json"
if not manifest_path.is_file():
    fail(f"{root} has no manifest.json -- not an FP4-CSF download (expected the "
         "GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD repository root with metadata/ and tensors/)")
raw = manifest_path.read_bytes()
try:
    manifest = json.loads(raw)
except ValueError as error:
    fail(f"manifest.json is not JSON: {error}")
schema = manifest.get("schema")
if schema != "lil-nvfp4-csf-checkpoint/1":
    fail(f"manifest schema {schema!r}; this launcher serves lil-nvfp4-csf-checkpoint/1")
if manifest.get("family") != "glm53_nvfp4":
    fail(f"manifest family {manifest.get('family')!r}; expected glm53_nvfp4 (GLM-5.3-Flash)")

missing = []
if not (root / "build-contract.json").is_file():
    missing.append("build-contract.json")
for name, digest in sorted(manifest.get("metadata_sha256", {}).items()):
    path = root / "metadata" / name
    if not path.is_file():
        missing.append(f"metadata/{name}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        missing.append(f"metadata/{name} (sha256 mismatch)")
shards = manifest.get("shards", [])
if not shards:
    fail("manifest lists no shards")
for shard in shards:
    path = root / "tensors" / shard["file"]
    size = shard.get("target_file_bytes")
    if not path.is_file():
        missing.append(f"tensors/{shard['file']}")
    elif size is not None and path.stat().st_size != size:
        missing.append(f"tensors/{shard['file']} ({path.stat().st_size} of {size} bytes)")
if missing:
    more = f" and {len(missing) - 6} more" if len(missing) > 6 else ""
    fail("incomplete or modified download (missing, partial, or sha256 mismatch): " + ", ".join(missing[:6]) + more)

config = json.loads((root / "metadata" / "config.json").read_text())
holder = config if "quantization_config" in config else config.get("text_config")
if not isinstance(holder, dict) or not isinstance(holder.get("quantization_config"), dict):
    fail("metadata/config.json has no quantization_config")
source = holder["quantization_config"]
if source.get("quant_method") == "nvfp4_csf":
    fail("metadata/config.json is already a serving config; point MODEL_HOST_PATH at the original download")
holder["quantization_config"] = {
    "quant_method": "nvfp4_csf",
    "format_version": 1,
    "checkpoint_root": container_root,
    "source_quantization_config": source,
}
arch = config.get("architectures") or []
if not any(a.startswith("Glm5Next") for a in arch):
    fail(f"architectures {arch}; expected Glm5Next* (GLM-5.3-Flash)")

identity = hashlib.sha256(b"lil-fp4-csf-v1\0" + raw).hexdigest()
name = re.sub(r"[^A-Za-z0-9._-]", "-", root.name) or "checkpoint"
serving = parent / f"{name}-{identity[:12]}"
tmp = pathlib.Path(tempfile.mkdtemp(prefix=".serving-", dir=parent))
for item in sorted((root / "metadata").iterdir()):
    if item.is_file() and item.name != "config.json":
        shutil.copyfile(item, tmp / item.name)
(tmp / "config.json").write_text(json.dumps(config, indent=2) + "\n")
os.chmod(tmp, 0o755)

def snapshot(d):
    return {f.name: f.read_bytes() for f in sorted(d.iterdir()) if f.is_file()}

if serving.is_dir() and snapshot(serving) == snapshot(tmp):
    # Unchanged: keep the directory a running container may have mounted.
    shutil.rmtree(tmp)
else:
    if serving.exists():
        shutil.rmtree(serving)
    tmp.rename(serving)
total = sum(s.get("target_file_bytes") or 0 for s in shards)
print(f"OK {identity} {serving} {len(shards)} {total}")
PYEOF
) || die "CSF checkpoint validation failed to run (python3 error above)"
case "$csf_out" in
  OK\ *) read -r _ csf_identity csf_serving_host csf_shards csf_bytes <<< "$csf_out" ;;
  ERROR\ *) die "MODEL_HOST_PATH: ${csf_out#ERROR }" ;;
  *) die "CSF checkpoint validation produced no result: $csf_out" ;;
esac
if [ -n "$CHECKPOINT_MANIFEST_SHA256" ]; then
  [ "$CHECKPOINT_MANIFEST_SHA256" = "$csf_identity" ] \
    || die "checkpoint identity $csf_identity != CHECKPOINT_MANIFEST_SHA256 $CHECKPOINT_MANIFEST_SHA256 -- the two nodes do not hold the same checkpoint"
fi

# ----------------------------------------------------------- sizes / graphs

for name in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS; do
  require_positive_integer "$name"
done
decode_batch=$(( MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1) ))
if [ "$MAX_CUDAGRAPH_CAPTURE_SIZE" = auto ]; then
  # The largest decode step: every slot verifying a full draft.
  MAX_CUDAGRAPH_CAPTURE_SIZE=$decode_batch
fi
require_positive_integer MAX_CUDAGRAPH_CAPTURE_SIZE
[ "$MAX_NUM_BATCHED_TOKENS" -ge "$decode_batch" ] \
  || die "MAX_NUM_BATCHED_TOKENS ($MAX_NUM_BATCHED_TOKENS) is below one speculative decode step ($decode_batch)"
[ "$decode_batch" -le "$MAX_CUDAGRAPH_CAPTURE_SIZE" ] \
  || warn "a full decode step is $decode_batch rows (MAX_NUM_SEQS x (NUM_SPECULATIVE_TOKENS+1)) but MAX_CUDAGRAPH_CAPTURE_SIZE=$MAX_CUDAGRAPH_CAPTURE_SIZE; the largest batches fall out of graph replay"

# Speculative decode only ever produces row counts that are multiples of
# (depth+1), plus 1 and 2 for single-token steps. auto captures exactly
# those (the release's TP2 list for MTP3 x 8 slots is 1,2,4,8,...,32);
# the engine's default grid would also capture odd sizes and pad C3/C5/...
# steps up to the next captured size. vLLM requires the list to end at
# --max-cudagraph-capture-size.
case "$CUDAGRAPH_CAPTURE_SIZES" in
  "") ;;
  auto)
    if [ "$NUM_SPECULATIVE_TOKENS" = 0 ]; then
      CUDAGRAPH_CAPTURE_SIZES=""
    else
      capture_step=$(( NUM_SPECULATIVE_TOKENS + 1 ))
      capture_list="1"
      [ "$MAX_CUDAGRAPH_CAPTURE_SIZE" -lt 2 ] || capture_list="$capture_list,2"
      capture_size=$capture_step
      while [ "$capture_size" -le "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]; do
        [ "$capture_size" -le 2 ] || capture_list="$capture_list,$capture_size"
        capture_size=$(( capture_size + capture_step ))
      done
      case ",$capture_list," in
        *",$MAX_CUDAGRAPH_CAPTURE_SIZE,"*) ;;
        *) capture_list="$capture_list,$MAX_CUDAGRAPH_CAPTURE_SIZE" ;;
      esac
      CUDAGRAPH_CAPTURE_SIZES=$capture_list
    fi
    ;;
  *)
    printf '%s' "$CUDAGRAPH_CAPTURE_SIZES" | grep -Eq '^[1-9][0-9]*(,[1-9][0-9]*)*$' \
      || die "CUDAGRAPH_CAPTURE_SIZES must be auto, empty, or ascending comma-separated positive integers (no spaces): $CUDAGRAPH_CAPTURE_SIZES"
    capture_prev=0
    IFS=, read -r -a capture_items <<< "$CUDAGRAPH_CAPTURE_SIZES"
    for capture_size in "${capture_items[@]}"; do
      [ "$capture_size" -gt "$capture_prev" ] || die "CUDAGRAPH_CAPTURE_SIZES must be strictly ascending: $CUDAGRAPH_CAPTURE_SIZES"
      capture_prev=$capture_size
    done
    [ "$capture_prev" = "$MAX_CUDAGRAPH_CAPTURE_SIZE" ] \
      || die "CUDAGRAPH_CAPTURE_SIZES must end at MAX_CUDAGRAPH_CAPTURE_SIZE ($MAX_CUDAGRAPH_CAPTURE_SIZE): $CUDAGRAPH_CAPTURE_SIZES"
    ;;
esac
capture_sizes_json=""
[ -z "$CUDAGRAPH_CAPTURE_SIZES" ] || capture_sizes_json=",\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]"

if [ "$MAX_MODEL_LEN" = -1 ]; then
  [ -n "$KV_CACHE_MEMORY_BYTES" ] \
    || die "MAX_MODEL_LEN=-1 (auto-fit to the KV pool) needs KV_CACHE_MEMORY_BYTES pinned"
else
  require_positive_integer MAX_MODEL_LEN
fi

case "$LANGUAGE_MODEL_ONLY" in 0|1) : ;; *) die "LANGUAGE_MODEL_ONLY must be 0 or 1: $LANGUAGE_MODEL_ONLY" ;; esac

require_port API_PORT
require_port MASTER_PORT
[ "$API_PORT" != "$MASTER_PORT" ] || die "API_PORT and MASTER_PORT must differ"

require_unit_fraction GPU_MEMORY_UTILIZATION
[ -z "$KV_CACHE_MEMORY_BYTES" ] || require_positive_integer KV_CACHE_MEMORY_BYTES
[ -n "$KV_CACHE_MEMORY_BYTES" ] \
  || warn "KV_CACHE_MEMORY_BYTES is empty: vLLM profiles the pool at GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION. Use that only for one measuring boot, then pin the reported bytes"

for name in ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL VLLM_ENABLE_PCIE_ALLREDUCE ENABLE_FLASHINFER_AUTOTUNE; do
  require_bool "$name"
done
[ "$ENABLE_PREFIX_CACHING" = 1 ] \
  || warn "ENABLE_PREFIX_CACHING=0: every multi-turn request re-prefills its whole history"

case "$KV_CACHE_DTYPE" in
  fp8|auto|fp8_ds_mla|nvfp4_ds_mla) ;;
  *) die "KV_CACHE_DTYPE must be fp8, auto, fp8_ds_mla, or nvfp4_ds_mla: $KV_CACHE_DTYPE" ;;
esac
case "$DFLASH_KV_CACHE_DTYPE" in fp8|auto) ;; *) die "DFLASH_KV_CACHE_DTYPE must be fp8 or auto: $DFLASH_KV_CACHE_DTYPE" ;; esac
case "$CUDAGRAPH_MODE" in
  FULL|FULL_AND_PIECEWISE|PIECEWISE|NONE) ;;
  *) die "CUDAGRAPH_MODE must be FULL, FULL_AND_PIECEWISE, PIECEWISE, or NONE: $CUDAGRAPH_MODE" ;;
esac

[ "$CUTE_DSL_ARCH" = sm_121a ] \
  || die "CUTE_DSL_ARCH must be sm_121a on DGX Spark (got $CUTE_DSL_ARCH); sm_120a is the RTX PRO 6000 value"
[ "$VLLM_ENABLE_PCIE_ALLREDUCE" = 0 ] \
  || die "VLLM_ENABLE_PCIE_ALLREDUCE must be 0 on a two-node pair (TP all-reduce crosses the CX7 link)"

case "$API_KEY" in *[[:space:]]*) die "API_KEY must not contain whitespace" ;; esac
case "$MOE_BACKEND" in b12x) : ;; *) die "MOE_BACKEND must be b12x: the NVFP4-CSF reader hands the routed experts to b12x in compressed form ($MOE_BACKEND)" ;; esac
case "$ATTENTION_BACKEND" in B12X|auto) : ;; *) die "ATTENTION_BACKEND must be B12X or auto: $ATTENTION_BACKEND" ;; esac
case "$LINEAR_BACKEND" in
  b12x) : ;;
  *) warn "LINEAR_BACKEND=$LINEAR_BACKEND: the MXFP8 attention/shared-expert projections are qualified on b12x; auto may pick FlashInfer kernels unqualified on sm_121" ;;
esac

# GB10 is capability 12.1. Two GLM-5.3 features remain gated to (12, 0):
#   VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4 raises at draft load ("requires CUDA
#     capability 12.0; got 12.1"); the GB10 draft head is VLLM_MTP_NVFP4_LM_HEAD=1.
#   VLLM_GLM53_L2_PREFETCH is off by default on 12.1; =1 forces it. The GLM
#     windows (20/50/15 MB) are sized for the 128 MB L2 of an RTX PRO 6000;
#     GB10 has 24 MB (upstream's GB10 DS4.1 prefetcher runs 20/16/20 MB with
#     a 4-CTA grid, which this pin already selects on 12.1).
case "${VLLM_GLM53_MTP_DRAFT_HEAD:-bf16}" in
  bf16) : ;;
  nvfp4) die "VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4 requires CUDA capability 12.0; GB10 is 12.1. Use VLLM_GLM53_MTP_DRAFT_HEAD=bf16 with VLLM_MTP_NVFP4_LM_HEAD=1" ;;
  *) die "VLLM_GLM53_MTP_DRAFT_HEAD must be bf16 or nvfp4: $VLLM_GLM53_MTP_DRAFT_HEAD" ;;
esac
if [ "${VLLM_GLM53_L2_PREFETCH:-}" = 1 ]; then
  [ -n "${VLLM_GLM53_L2_PREFETCH_BUDGET_B_MB:-}" ] \
    || warn "VLLM_GLM53_L2_PREFETCH=1 with the default 50 MB window B: larger than GB10's 24 MB L2. Set VLLM_GLM53_L2_PREFETCH_BUDGET_{A,B,C,A_MLA}_MB (e.g. 20/16/15/16) for the A/B"
fi
# Unified memory: pinned host RAM IS the GPU's memory on GB10, so the
# release's embedding-in-host-RAM saving does not exist here; it only adds
# the UVA read cost (prefill -1 to -3% upstream).
[ "${VLLM_GLM53_EMBED_HOST:-0}" != 1 ] \
  || warn "VLLM_GLM53_EMBED_HOST=1 frees nothing on GB10 (host RAM and GPU memory are the same pool) and costs 1-3% prefill; set 0"

[ "$NODE_RANK" != 0 ] || [ "$MASTER_ADDR" = "$VLLM_HOST_IP" ] \
  || die "rank-0 MASTER_ADDR must equal rank-0 VLLM_HOST_IP"
[ "$NODE_RANK" != 1 ] || [ "$MASTER_ADDR" != "$VLLM_HOST_IP" ] \
  || die "rank-1 VLLM_HOST_IP must differ from MASTER_ADDR"

# Memory preflight. On this vLLM the startup gate (request_memory) compares
# total x GPU_MEMORY_UTILIZATION with the device's free memory, and on an
# integrated GPU (GB10) "free" is psutil's MemAvailable -- reclaimable page
# cache counts as free. The gate applies with or without the KV pin.
case "$MEM_PREFLIGHT" in die|warn|off) : ;; *) die "MEM_PREFLIGHT must be die, warn, or off" ;; esac
if [ "$MEM_PREFLIGHT" != off ] && [ -r /proc/meminfo ]; then
  mem_total_kib=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
  mem_avail_kib=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  required_kib=$(awk -v t="$mem_total_kib" -v u="$GPU_MEMORY_UTILIZATION" 'BEGIN{printf "%d", t*u}')
  if [ "$mem_avail_kib" -lt "$required_kib" ]; then
    stale="$("$CONTAINER_RUNTIME" ps --format '{{.Names}}' 2>/dev/null | grep -c '^glm53-flash-r' || true)"
    msg="MemAvailable $(awk -v a="$mem_avail_kib" 'BEGIN{printf "%.1f", a/1048576}') GiB < $(awk -v r="$required_kib" 'BEGIN{printf "%.1f", r/1048576}') GiB (MemTotal x GPU_MEMORY_UTILIZATION $GPU_MEMORY_UTILIZATION): vLLM refuses at startup"
    [ "${stale:-0}" = 0 ] || msg="$msg -- a glm53-flash container is already running on this host; --down it first"
    if [ "$MEM_PREFLIGHT" = warn ]; then warn "$msg"; else die "$msg"; fi
  fi
fi

# MASTER_ADDR must sit on a directly connected subnet; a typo that routes
# through a gateway hangs the torch.distributed rendezvous forever.
if command -v ip >/dev/null 2>&1; then
  master_route="$(ip route get "$MASTER_ADDR" 2>/dev/null | head -1)"
  case "$master_route" in
    "")        die "MASTER_ADDR=$MASTER_ADDR has no route from this host -- typo?" ;;
    *" via "*) die "MASTER_ADDR=$MASTER_ADDR is not on a directly connected subnet (route: $master_route) -- it must be rank 0's address on the CX7 link" ;;
  esac
  ip -4 -o addr show 2>/dev/null | grep -qw "$VLLM_HOST_IP" \
    || die "VLLM_HOST_IP ($VLLM_HOST_IP) is not an IPv4 address on this host; wrong rank's env file?"
fi

# ------------------------------------------------------------ fabric checks

[ "$NCCL_SOCKET_IFNAME" = "$GLOO_SOCKET_IFNAME" ] \
  || die "NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME must match on a pair"
[ "$NCCL_NET" = IB ] || die "NCCL_NET must be IB (RoCE presents IB semantics over Ethernet)"
[ "$NCCL_NET_PLUGIN" = none ] || die "NCCL_NET_PLUGIN must be none"
[ "$NCCL_TUNER_PLUGIN" = none ] || die "NCCL_TUNER_PLUGIN must be none"
[ "$NCCL_IB_DISABLE" = 0 ] || die "NCCL_IB_DISABLE must be 0 (the image bakes 1 for single-node use; without the override the pair falls back to TCP)"
#   single   : ONE cable, ONE PCIe function. MERGE_NICS=0, one device.
#   dualpath : ONE cable, BOTH PCIe functions of the cabled cage (the GB10
#              CX-7 is two Gen5 x4 functions behind one port). Each needs its
#              own IPv4 in its own /24 so its RoCEv2 GID row exists. 0/0, as
#              both published TP2 references on this cabling run it (el8:
#              0 required for separate QP setup). ~190 Gb/s measured here.
#   dual     : BOTH cages cabled. MERGE_NICS=1, SUBNET_AWARE_ROUTING=1.
case "$FABRIC_PROFILE" in single|dualpath|dual) : ;; *) die "FABRIC_PROFILE must be single, dualpath, or dual: $FABRIC_PROFILE" ;; esac
fabric_devices=$(printf '%s' "$NCCL_IB_HCA" | tr ',' '\n' | sed 's/^[=^]*//; s/:.*$//' | grep . || true)
fabric_count=$(printf '%s\n' "$fabric_devices" | grep -c . || true)
case "$FABRIC_PROFILE" in
  single)
    [ "$fabric_count" = 1 ] || die "FABRIC_PROFILE=single names exactly one RoCE device; got $NCCL_IB_HCA (dualpath for both functions of one cable)"
    [ "$NCCL_IB_MERGE_NICS" = 0 ] || die "FABRIC_PROFILE=single requires NCCL_IB_MERGE_NICS=0"
    ;;
  dualpath)
    [ "$fabric_count" = 2 ] || die "FABRIC_PROFILE=dualpath needs exactly the two PCIe functions of the cabled cage in NCCL_IB_HCA (e.g. rocep1s0f0,roceP2p1s0f0), got: $NCCL_IB_HCA"
    [ "$NCCL_IB_MERGE_NICS" = 0 ] \
      || warn "FABRIC_PROFILE=dualpath with NCCL_IB_MERGE_NICS=1: the published references on this cabling run 0 (el8: required for separate QP setup); A/B only"
    [ "$NCCL_IB_SUBNET_AWARE_ROUTING" = 0 ] \
      || warn "FABRIC_PROFILE=dualpath with NCCL_IB_SUBNET_AWARE_ROUTING=1: only matters behind a switch"
    ;;
  dual)
    [ "$fabric_count" = 2 ] || die "FABRIC_PROFILE=dual needs both rails in NCCL_IB_HCA, got: $NCCL_IB_HCA"
    [ "$NCCL_IB_MERGE_NICS" = 1 ] || die "FABRIC_PROFILE=dual requires NCCL_IB_MERGE_NICS=1"
    [ "$NCCL_IB_SUBNET_AWARE_ROUTING" = 1 ] || die "FABRIC_PROFILE=dual requires NCCL_IB_SUBNET_AWARE_ROUTING=1"
    ;;
esac
[ "$NCCL_PROTO" = LL,LL128,Simple ] || die "NCCL_PROTO must be LL,LL128,Simple"
[ "$NCCL_P2P_LEVEL" = SYS ] || die "NCCL_P2P_LEVEL must be SYS"
[ "$NCCL_CROSS_NIC" = 1 ] || die "NCCL_CROSS_NIC must be 1"
[ "$NCCL_CUMEM_ENABLE" = 0 ] || die "NCCL_CUMEM_ENABLE must be 0 (conflicts with expandable_segments)"
[ "$NCCL_IGNORE_CPU_AFFINITY" = 1 ] || die "NCCL_IGNORE_CPU_AFFINITY must be 1"
case "$NCCL_IB_GID_INDEX" in ''|*[!0-9]*) die "NCCL_IB_GID_INDEX must be a decimal integer" ;; esac

# Every listed RDMA device must exist here and carry a RoCEv2 IPv4 GID at
# NCCL_IB_GID_INDEX (RoCEnante uses the same index). A missing row is the
# classic silent hang: the first collective never completes.
if [ -d /sys/class/infiniband ]; then
  for fabric_dev in $fabric_devices; do
    fabric_sys=/sys/class/infiniband/$fabric_dev/ports/1
    [ -d "$fabric_sys" ] || die "RDMA device $fabric_dev (NCCL_IB_HCA) does not exist on this host (ibv_devices)"
    fabric_gid=$(cat "$fabric_sys/gids/$NCCL_IB_GID_INDEX" 2>/dev/null || true)
    fabric_type=$(cat "$fabric_sys/gid_attrs/types/$NCCL_IB_GID_INDEX" 2>/dev/null || true)
    case "$fabric_gid" in
      0000:0000:0000:0000:0000:ffff:*) ;;
      *) die "$fabric_dev has no IPv4 GID at index $NCCL_IB_GID_INDEX (got '${fabric_gid:-none}'): give its netdev an IPv4 in its own /24, then re-check with show_gids" ;;
    esac
    case "$fabric_type" in *v2*) ;; *) die "$fabric_dev GID index $NCCL_IB_GID_INDEX is '${fabric_type:-unknown}', not RoCE v2" ;; esac
  done
fi
[ -e /dev/infiniband ] || die "/dev/infiniband does not exist; the RDMA stack is not up (lsmod | grep mlx5_ib)"

if [ -n "$B12X_ROCE_HCA" ]; then
  roce_devices=$(printf '%s' "$B12X_ROCE_HCA" | tr ',' '\n' | sed 's/^[=^]*//; s/:.*$//' | grep . | LC_ALL=C sort | tr '\n' ' ')
  nccl_devices=$(printf '%s\n' "$fabric_devices" | LC_ALL=C sort | tr '\n' ' ')
  [ "$roce_devices" = "$nccl_devices" ] \
    || die "B12X_ROCE_HCA ($B12X_ROCE_HCA) must name the same devices as NCCL_IB_HCA ($NCCL_IB_HCA), or be empty to follow it"
fi
if [ "${VLLM_ENABLE_ROCE_ALLREDUCE:-0}" = 1 ]; then
  for name in VLLM_ROCE_ALLREDUCE_MAX_SIZE VLLM_ROCE_ALLGATHER_MAX_SIZE; do
    [ -z "${!name-}" ] || require_positive_integer "$name"
  done
  case "${B12X_ROCE_CACHE_DIR-}" in
    /cache*) ;;
    *) warn "B12X_ROCE_CACHE_DIR=${B12X_ROCE_CACHE_DIR:-<unset>} is outside the mounted /cache; the RoCE proxy recompiles every boot" ;;
  esac
  # GLM-5.3 hidden = 4096: an all-reduce of N tokens is N x 8 KiB (BF16).
  roce_limit=${VLLM_ROCE_ALLREDUCE_MAX_SIZE:-2097152}
  case "$roce_limit" in *[!0-9]*) roce_limit=2097152 ;; esac
  roce_desc="RoCEnante all-reduce <= $(( roce_limit / 1048576 )) MiB = $(( roce_limit / 8192 )) tokens; a full ${MAX_NUM_BATCHED_TOKENS}-token prefill step ($(( MAX_NUM_BATCHED_TOKENS * 8192 / 1048576 )) MiB) goes to $([ "$roce_limit" -ge $(( MAX_NUM_BATCHED_TOKENS * 8192 )) ] && echo RoCEnante || echo NCCL)"
else
  roce_desc="off (NCCL for every collective)"
fi

case ":$LD_PRELOAD:" in
  *":$VLLM_NCCL_SO_PATH:"*) ;;
  *) die "LD_PRELOAD must include VLLM_NCCL_SO_PATH ($VLLM_NCCL_SO_PATH)" ;;
esac
case ":$LD_PRELOAD:" in
  *"/cuda/compat/lib.real/"*) warn "LD_PRELOAD carries the NGC forward-compat libcuda (compat/lib.real); on this pair that crashed torch's CUDA init (cu133 images). The cu132 image needs /usr/local/cuda/compat/libcuda.so.1" ;;
esac

# ------------------------------------------------------- host / image checks

port_in_use() {
  command -v ss >/dev/null 2>&1 || return 1
  ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"
}
if [ "$NODE_RANK" = 0 ] && [ "$mode" = --run ]; then
  ! port_in_use "$API_PORT" || die "API_PORT $API_PORT is already listening on this host; --down the old container or pick another port"
  ! port_in_use "$MASTER_PORT" || die "MASTER_PORT $MASTER_PORT is already listening on this host"
fi

if [ -d /dev/shm ]; then
  shm_mb=$(df -Pm /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
  case "$shm_mb" in
    ''|*[!0-9]*) ;;
    *) [ "$shm_mb" -ge 8192 ] || warn "/dev/shm is ${shm_mb} MiB; with --ipc host this is the real limit and vLLM's shm_broadcast can stall below ~8 GiB" ;;
  esac
fi

# Image probes (only when the image is present locally): LD_PRELOAD paths,
# the paired NVFP4-CSF support in vLLM AND b12x, and fastokens.
image_desc="not present locally (probes skipped)"
if [ "${SKIP_IMAGE_CHECKS:-0}" != 1 ] \
   && command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1 \
   && "$CONTAINER_RUNTIME" image inspect "$SERVING_IMAGE" >/dev/null 2>&1; then
  preload_missing=""
  old_ifs=$IFS; IFS=:
  for lib in $LD_PRELOAD; do
    [ -n "$lib" ] || continue
    "$CONTAINER_RUNTIME" run --rm --entrypoint test "$SERVING_IMAGE" -f "$lib" >/dev/null 2>&1 \
      || preload_missing="$preload_missing $lib"
  done
  IFS=$old_ifs
  [ -z "$preload_missing" ] \
    || die "LD_PRELOAD names path(s) absent from the image:$preload_missing -- every process in the container would fail at exec"

  # Source probe, not imports: the loader registries pull in modules that
  # need a GPU driver at import time.
  csf_probe=$("$CONTAINER_RUNTIME" run --rm -i --entrypoint /opt/venv/bin/python "$SERVING_IMAGE" - <<'PY' 2>/dev/null || true
import importlib.metadata as md, importlib.util, pathlib
def src(mod):
    spec = importlib.util.find_spec(mod)
    return pathlib.Path(spec.origin).read_text() if spec and spec.origin else ""
q = src("vllm.model_executor.layers.quantization")
l = src("vllm.model_executor.model_loader")
v = '"nvfp4_csf"' in q and '"nvfp4_csf": Nvfp4CsfModelLoader' in l
root = pathlib.Path(importlib.util.find_spec("b12x").origin).parent
b12x_src = "".join(f.read_text(errors="ignore") for f in (root / "moe").rglob("*.py"))
b = all(f"class {n}" in b12x_src for n in ("Nvfp4CsfWeights", "CsfScalePlanes"))
a4 = (root / "moe" / "_shared" / "kernels" / "w4a16" / "prefill_a4.py").is_file()
print("CSF", int(v), int(b), int(a4), md.version("nvidia-cutlass-dsl"))
PY
)
  case "$csf_probe" in
    "CSF 1 1 "*) : ;;
    "CSF "*) die "$SERVING_IMAGE cannot serve NVFP4-CSF (vllm reader/b12x CSF weights: ${csf_probe#CSF }). Build build-kk-integration-cache-cu132.env (vLLM f1c2508f + b12x 52640cb1)" ;;
    *) die "image capability probe failed to run in $SERVING_IMAGE (output: ${csf_probe:-none}); SKIP_IMAGE_CHECKS=1 to bypass" ;;
  esac
  read -r _ _ _ csf_a4 csf_dsl <<< "$csf_probe"
  [ "$PREFILL_ACTIVATIONS" != a4 ] || [ "$csf_a4" = 1 ] \
    || die "PREFILL_ACTIVATIONS=a4 but the image's b12x has no W4A16 A4 prefill kernels; set PREFILL_ACTIVATIONS=a16 or rebuild"
  image_desc="NVFP4-CSF reader + b12x CSF + A4 prefill present; CuTe DSL $csf_dsl"

  if [ "$VLLM_USE_FASTOKENS" = 1 ]; then
    fk_label=$("$CONTAINER_RUNTIME" image inspect --format '{{index .Config.Labels "org.local-inference.fastokens.version"}}' "$SERVING_IMAGE" 2>/dev/null || true)
    [ -n "$fk_label" ] && [ "$fk_label" != "<no value>" ] \
      || die "VLLM_USE_FASTOKENS=1 but $SERVING_IMAGE carries no fastokens (no org.local-inference.fastokens.version label); vLLM would stop at tokenizer load. Rebuild with PATCH_FASTOKENS=on or set 0"
    image_desc="$image_desc; fastokens $fk_label"
  fi
fi

# ----------------------------------------------------------- launch command

container_name="$mgmt_container"
dflash_container_path=/models/glm-5.3-flash-dflash2
chat_template_container_path=/models/chat_template.jinja

# The checkpoint ships the official template; the release serves a variant
# that renders assistant content exactly as generated (no strip), so a
# returned answer re-renders to the same tokens and the next turn reuses its
# prefix checkpoint instead of re-prefilling the reply. Relative paths are
# resolved against this script's folder.
if [ -n "$CHAT_TEMPLATE_HOST_PATH" ]; then
  case "$CHAT_TEMPLATE_HOST_PATH" in /*) ;; *) CHAT_TEMPLATE_HOST_PATH="$script_dir/$CHAT_TEMPLATE_HOST_PATH" ;; esac
  [ -f "$CHAT_TEMPLATE_HOST_PATH" ] || die "CHAT_TEMPLATE_HOST_PATH does not exist: $CHAT_TEMPLATE_HOST_PATH"
fi

extra_args=(--prefill-schedule-interval "$PREFILL_SCHEDULE_INTERVAL")
[ -z "$COMPILATION_LEVEL" ] || extra_args+=("-O$COMPILATION_LEVEL")
[ -z "$KDA_PREFILL_BACKEND" ] || extra_args+=(--kda-prefill-backend "$KDA_PREFILL_BACKEND")
[ -z "$GDN_DECODE_KERNEL" ] || extra_args+=(--gdn-decode-kernel "$GDN_DECODE_KERNEL")
[ -z "$KDA_DECODE_BACKEND" ] || extra_args+=(--kda-decode-backend "$KDA_DECODE_BACKEND")
case "$REPLAYSSM" in
  0) extra_args+=(--no-use-replayssm) ;;
  1) extra_args+=(--use-replayssm) ;;
esac
[ -z "$PREFILL_COMPUTE_SHARE" ] || extra_args+=(--prefill-compute-share "$PREFILL_COMPUTE_SHARE")
[ -z "$PREFILL_COMPUTE_HALF_LIFE" ] || extra_args+=(--prefill-compute-half-life "$PREFILL_COMPUTE_HALF_LIFE")
[ -z "$MAX_PARALLEL_PREFILLS" ] || extra_args+=(--max-parallel-prefills "$MAX_PARALLEL_PREFILLS")
[ -z "$RECURRENT_CHECKPOINT_POLICY" ] || extra_args+=(--recurrent-checkpoint-policy "$RECURRENT_CHECKPOINT_POLICY")
# Sampling and chat-template defaults are built HERE (JSON in a sourced env
# file loses its quotes). Per-request fields still override them.
sampling_json=""
[ -z "$SAMPLING_TEMPERATURE" ] || sampling_json="\"temperature\":$SAMPLING_TEMPERATURE"
[ -z "$SAMPLING_TOP_P" ] || sampling_json="${sampling_json:+$sampling_json,}\"top_p\":$SAMPLING_TOP_P"
[ -z "$SAMPLING_TOP_K" ] || sampling_json="${sampling_json:+$sampling_json,}\"top_k\":$SAMPLING_TOP_K"
[ -z "$SAMPLING_MIN_P" ] || sampling_json="${sampling_json:+$sampling_json,}\"min_p\":$SAMPLING_MIN_P"
[ -z "$SAMPLING_REPETITION_PENALTY" ] || sampling_json="${sampling_json:+$sampling_json,}\"repetition_penalty\":$SAMPLING_REPETITION_PENALTY"
[ -z "$sampling_json" ] || extra_args+=(--override-generation-config "{$sampling_json}")
template_json=""
[ -z "$REASONING_EFFORT" ] || template_json="\"reasoning_effort\":\"$REASONING_EFFORT\""
if [ -n "$CLEAR_THINKING" ]; then
  ct=false; [ "$CLEAR_THINKING" = 1 ] && ct=true
  template_json="${template_json:+$template_json,}\"clear_thinking\":$ct"
fi
[ -z "$template_json" ] || extra_args+=(--default-chat-template-kwargs "{$template_json}")
[ -z "$MM_PROCESSOR_CACHE_GB" ] || extra_args+=(--mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB")
[ -z "$MM_ENCODER_TP_MODE" ] || extra_args+=(--mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")
[ "$GENERATION_CONFIG" = auto ] || extra_args+=(--generation-config "$GENERATION_CONFIG")
[ -z "$CHAT_TEMPLATE_HOST_PATH" ] || extra_args+=(--chat-template "$chat_template_container_path")

lm_only_args=()
if [ "$LANGUAGE_MODEL_ONLY" = 1 ]; then
  lm_only_args=(--language-model-only)
else
  require_bool LIMIT_MM
  if [ "$LIMIT_MM" = 1 ]; then
    require_nonnegative_integer MM_IMAGES
    require_nonnegative_integer MM_VIDEOS
    lm_only_args=(--limit-mm-per-prompt "$(printf '{"image":%d,"video":%d}' "$MM_IMAGES" "$MM_VIDEOS")")
  fi
fi

speculative_args=()
if [ "$NUM_SPECULATIVE_TOKENS" -gt 0 ]; then
  case "$SPECULATOR" in
    mtp)
      # The MTP layer is read from the target checkpoint (no "model"), so it
      # follows the CSF serving directory and load format automatically.
      adaptive_fields=
      if [ "$ADAPTIVE_SPECULATIVE_TOKENS" = 1 ]; then
        adaptive_fields=$(printf ',"adaptive_speculative_tokens_window":%s,"adaptive_speculative_tokens_initial":%s' \
          "$ADAPTIVE_SPECULATIVE_TOKENS_WINDOW" "$ADAPTIVE_SPECULATIVE_TOKENS_INITIAL")
      fi
      speculative_config=$(printf \
        '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"%s","rejection_sample_method":"%s","moe_backend":"%s","attention_backend":"%s"%s}' \
        "$NUM_SPECULATIVE_TOKENS" "$DRAFT_SAMPLE_METHOD" "$REJECTION_SAMPLE_METHOD" "$MTP_MOE_BACKEND" "$MTP_ATTENTION_BACKEND" "$adaptive_fields")
      ;;
    dflash2)
      # The DFlash2 drafter is a plain safetensors checkpoint: without
      # draft_load_config it inherits the target's nvfp4_csf load format and
      # fails. The release profile pins load_format auto for exactly this.
      dflash_attention_json=
      [ -z "$DFLASH_ATTENTION_BACKEND" ] || dflash_attention_json=$(printf ',"attention_backend":"%s"' "$DFLASH_ATTENTION_BACKEND")
      speculative_config=$(printf \
        '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"kv_cache_dtype":"%s","draft_sample_method":"%s","rejection_sample_method":"%s","draft_load_config":{"load_format":"auto"}%s}' \
        "$dflash_container_path" "$NUM_SPECULATIVE_TOKENS" "$DFLASH_KV_CACHE_DTYPE" "$DRAFT_SAMPLE_METHOD" "$REJECTION_SAMPLE_METHOD" "$dflash_attention_json")
      ;;
  esac
  speculative_args=(--speculative-config "$speculative_config")
fi

compilation_config=$(printf '{"cudagraph_mode":"%s","custom_ops":["all"]%s}' "$CUDAGRAPH_MODE" "$capture_sizes_json")

case "$CONTAINER_RUNTIME" in
  podman) gpu_args=(--device nvidia.com/gpu=all --security-opt label=disable) ;;
  *)      gpu_args=(--gpus all) ;;
esac

mount_args=(
  -v "$MODEL_HOST_PATH:$model_container_path:ro"
  -v "$csf_serving_host:$serving_container_path:ro"
  -v "$CACHE_HOST_PATH:/cache"
)
[ "$SPECULATOR" != dflash2 ] || mount_args+=(-v "$DFLASH_MODEL_HOST_PATH:$dflash_container_path:ro")
[ -z "$CHAT_TEMPLATE_HOST_PATH" ] || mount_args+=(-v "$CHAT_TEMPLATE_HOST_PATH:$chat_template_container_path:ro")
if [ -n "$TORCH_PROFILE_HOST_DIR" ]; then
  mkdir -p "$TORCH_PROFILE_HOST_DIR"
  mount_args+=(-v "$TORCH_PROFILE_HOST_DIR:/profiles")
fi

memory_args=()
if [ -n "$CONTAINER_MEMORY_GB" ]; then
  require_positive_integer CONTAINER_MEMORY_GB
  memory_args=(--memory "${CONTAINER_MEMORY_GB}g" --memory-swap "$(( CONTAINER_MEMORY_GB + 4 ))g")
fi
profile_env_args=()
[ -z "$TORCH_PROFILE_HOST_DIR" ] || profile_env_args=(-e VLLM_TORCH_PROFILER_DIR=/profiles)

command=(
  "$CONTAINER_RUNTIME" run -d
  --name "$container_name"
  --pull never
  --network host
  --ipc host
  --shm-size "$SHM_SIZE"
  "${gpu_args[@]}"
  --ulimit memlock=-1:-1
  "${memory_args[@]}"
  --device /dev/infiniband
  "${mount_args[@]}"
  --env-file "$env_file"
  # The image bakes VLLM_PCIE_ALLREDUCE_BACKEND=cpp (pre-rename); vLLM
  # validates it with choices ['b12x'] at worker start even when the PCIe
  # path is off. -e beats both --env-file and the baked ENV.
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
  "${precision_env_args[@]}"
  "${dcp_env_args[@]}"
  "${profile_env_args[@]}"
  --entrypoint /opt/venv/bin/vllm
  "$SERVING_IMAGE"
  serve "$serving_container_path"

  # --- topology -----------------------------------------------------------
  --tensor-parallel-size 2
  --nnodes 2
  --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --distributed-executor-backend mp
  --pipeline-parallel-size 1
  "${dcp_args[@]}"

  # --- memory -------------------------------------------------------------
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --block-size "$BLOCK_SIZE"

  # --- model / kernels (FP4-CSF: the reader owns the routed experts) -------
  --dtype bfloat16
  --quantization nvfp4_csf
  --load-format nvfp4_csf
  --attention-backend "$ATTENTION_BACKEND"
  --moe-backend "$MOE_BACKEND"
  --linear-backend "$LINEAR_BACKEND"
  "${lm_only_args[@]}"
  --mamba-cache-mode align
  --compilation-config "$compilation_config"
  --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"

  # --- scheduling ---------------------------------------------------------
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  "${extra_args[@]}"

  # --- model behaviour ----------------------------------------------------
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
  "${speculative_args[@]}"
  --served-model-name "$SERVED_MODEL_NAME"

  # --- observability ------------------------------------------------------
  --enable-prompt-tokens-details
  --enable-force-include-usage
  --enable-request-id-headers
)

if [ "$ENABLE_PREFIX_CACHING" = 1 ]; then command+=(--enable-prefix-caching); else command+=(--no-enable-prefix-caching); fi
if [ "$ENABLE_CHUNKED_PREFILL" = 1 ]; then command+=(--enable-chunked-prefill); fi
if [ "$ENABLE_FLASHINFER_AUTOTUNE" = 1 ]; then command+=(--enable-flashinfer-autotune); else command+=(--no-enable-flashinfer-autotune); fi
[ -z "$KV_CACHE_MEMORY_BYTES" ] || command+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")

if [ "$NODE_RANK" = 0 ]; then
  command+=(--host 0.0.0.0 --port "$API_PORT")
  [ -z "$API_KEY" ] || command+=(--api-key "$API_KEY")
else
  command+=(--headless)
fi

[ "${#passthrough[@]}" -eq 0 ] || command+=("${passthrough[@]}")

# ------------------------------------------------------------------ output

printf "Local rank input checks passed.\n"
printf '  rank:                    %s\n' "$NODE_RANK"
printf '  checkpoint:              %s (%s shards, %s GB)\n' "$MODEL_HOST_PATH" "$csf_shards" "$(awk -v b="$csf_bytes" 'BEGIN{printf "%.1f", b/1e9}')"
printf '  checkpoint identity:     %s (must match on both ranks)\n' "$csf_identity"
printf '  serving dir:             %s -> %s\n' "$csf_serving_host" "$serving_container_path"
[ "$SPECULATOR" != dflash2 ] || printf '  draft model:             %s\n' "$DFLASH_MODEL_HOST_PATH"
printf '  image:                   %s (%s)\n' "$SERVING_IMAGE" "$image_desc"
printf '  precision:               %s\n' "$precision_desc"
printf '  parallelism:             TP2 x DCP%s\n' "$DCP_SIZE"
printf '  SPECULATOR:              %s (%s draft tokens, %s / %s%s)\n' "$SPECULATOR" "$NUM_SPECULATIVE_TOKENS" "$DRAFT_SAMPLE_METHOD" "$REJECTION_SAMPLE_METHOD" \
  "$([ "$SPECULATOR" = mtp ] && printf ', MoE %s' "$MTP_MOE_BACKEND")"
printf '  MAX_MODEL_LEN:           %s\n' "$MAX_MODEL_LEN"
printf '  MAX_NUM_SEQS:            %s (decode step %s rows)\n' "$MAX_NUM_SEQS" "$decode_batch"
printf '  MAX_NUM_BATCHED_TOKENS:  %s\n' "$MAX_NUM_BATCHED_TOKENS"
if [ -n "$PREFILL_COMPUTE_SHARE" ]; then
  printf '  PREFILL:                 compute share %s, interval %s%s\n' "$PREFILL_COMPUTE_SHARE" "$PREFILL_SCHEDULE_INTERVAL" "${MAX_PARALLEL_PREFILLS:+, max parallel $MAX_PARALLEL_PREFILLS}"
else
  printf '  PREFILL:                 schedule interval %s%s\n' "$PREFILL_SCHEDULE_INTERVAL" "${MAX_PARALLEL_PREFILLS:+, max parallel $MAX_PARALLEL_PREFILLS}"
fi
printf '  KV_CACHE_MEMORY_BYTES:   %s (%s)\n' "${KV_CACHE_MEMORY_BYTES:-profiled at $GPU_MEMORY_UTILIZATION}" "$KV_CACHE_DTYPE"
printf '  GPU_MEMORY_UTILIZATION:  %s (startup gate%s)\n' "$GPU_MEMORY_UTILIZATION" "$([ -n "$KV_CACHE_MEMORY_BYTES" ] && echo '; the pin sizes the pool')"
printf '  CUDAGRAPH_MODE:          %s (capture max %s: %s)\n' "$CUDAGRAPH_MODE" "$MAX_CUDAGRAPH_CAPTURE_SIZE" "${CUDAGRAPH_CAPTURE_SIZES:-engine default grid}"
printf '  prefix cache:            GPU only (%s)\n' "${RECURRENT_CHECKPOINT_POLICY:-auto}"
printf '  fabric profile:          %s (NCCL %s; RoCEnante %s)\n' "$FABRIC_PROFILE" "$NCCL_IB_HCA" "${B12X_ROCE_HCA:-follows NCCL_IB_HCA}"
printf '  collectives:             %s\n' "$roce_desc"
printf '  tokenizer backend:       %s\n' "$([ "$VLLM_USE_FASTOKENS" = 1 ] && echo fastokens || echo 'HF tokenizers (standard)')"
printf '  chat template:           %s\n' "${CHAT_TEMPLATE_HOST_PATH:-checkpoint default}"
printf '  API_KEY:                 %s\n' "$([ -n "$API_KEY" ] && echo 'set (Bearer required)' || echo 'none (open port)')"
printf '  command:'
printf ' %q' "${command[@]}"
printf '\n'

[ "$mode" = --run ] || exit 0

# --------------------------------------------------------------------- run

require_runtime
"$CONTAINER_RUNTIME" image inspect "$SERVING_IMAGE" >/dev/null 2>&1 \
  || die "pinned image is not present; build or load it before launching: $SERVING_IMAGE"
if "$CONTAINER_RUNTIME" container inspect "$container_name" >/dev/null 2>&1; then
  die "container already exists; remove it intentionally before relaunch: $container_name"
fi

if [ "$fresh_follow" = 1 ]; then
  "${command[@]}"
  printf '%s started; following logs (Ctrl-C detaches, the container keeps running)\n' "$container_name"
  exec "$CONTAINER_RUNTIME" logs -f "$container_name"
else
  exec "${command[@]}"
fi
