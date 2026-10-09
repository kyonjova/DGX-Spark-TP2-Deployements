#!/usr/bin/env bash
# glm-53-flash-csf_pair-serve.sh
#
# Validate or start one rank of a two-node DGX Spark pair serving
# local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD (NVFP4 QAD routed
# experts with losslessly compressed scales, MXFP8 attention and shared
# experts, NVFP4 MTP experts) at TP=2 under a karmic-kraken image built by
# dgx-spark-builder/build-spark-cache-cu132.sh with
# build-kk-integration-cache-cu132.env (local/vllm:karmic-kraken-intcache1008-cu132:
# vLLM integration/karmic-kraken-beta 19f2c20e on FlashInfer 0.7.1 with B12X
# inside). Only that image line is served: --check probes the image and
# refuses one without the ModelOpt-recipe CSF reader.
#
# What this launcher adds over a plain vllm serve:
#   * The Hugging Face-layout FP4-CSF checkpoint (HF main from dec48abd):
#     config.json recipes mark the CSF expert scales and vLLM reads the
#     download directory as is (--quantization modelopt_mixed). --check
#     validates recipes, index and shards, computes LIL's content identity
#     (CHECKPOINT_IDENTITY_SHA256) and the download revision
#     (CHECKPOINT_REVISION), and counts the calibrated expert input scales
#     A4 prefill needs.
#   * LOAD_FORMAT (env section 7; default b12x): vLLM's native b12x loader
#     (O_DIRECT reads through io_uring; no VLLM_PLUGINS entry) with the
#     io_uring seccomp block shared by every *_pair-serve.sh; instanttensor
#     (+ its INSTANTTENSOR_* staging bounds), fastsafetensors, safetensors
#     and auto are the alternatives.
#   * Expert precision as three named choices (EXPERT_ACTIVATIONS,
#     ROUTER_WEIGHTS, PREFILL_ACTIVATIONS) mapped to the b12x/vLLM variables
#     the same way the release launcher maps them. The env file must not set
#     the raw variables; the launcher passes them with -e.
#   * Optional decode context parallelism (DCP_SIZE=2) with the derived
#     interleave and CKV-gather settings.
#   * FABRIC_PROFILE (single | dualpath | dual) with per-device RoCEv2 GID
#     checks.
#   * CUDAGRAPH_CAPTURE_SIZES / MAX_CUDAGRAPH_CAPTURE_SIZE auto: capture
#     exactly the decode shapes MTP/DFlash produce.
#   * External prefix cache (env section 9, PREFIX_CACHE=none|sparkcache|
#     lmcache): builds --kv-transfer-config, mounts PREFIX_CACHE_HOST_PATH at
#     /prefix-cache, and for lmcache starts a CPU-only sidecar
#     (<name>-lmcache) before vLLM and waits for its health check. The cache
#     identity defaults to the checkpoint identity, the namespace includes
#     DCP_SIZE, and REPLAYSSM is forced to 0 (1 is refused) whenever a
#     connector is on. Default none.
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

  --check     validate the env file, image and checkpoint, print the launch command (default)
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

# Reclaim page cache. The startup gate reads MemAvailable on GB10 (integrated
# GPU -> psutil), so page cache does not block it, and the b12x loader reads
# with O_DIRECT (no page cache). --fresh is for comparable benchmark boots:
# every boot starts from the same reclaimed state.
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
           B12X_W4A16_A4_PREFILL VLLM_B12X_NVFP4_ACTIVATION_MODE \
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
  # PREFIX_CACHE=lmcache sidecar (this launcher's or the glm-5.3-flash-cache
  # launcher's): stopped AFTER vLLM so in-flight stores and restores drain
  # into a live server. Removed whatever PREFIX_CACHE says now -- it holds
  # /dev/shm and host RAM.
  if "$CONTAINER_RUNTIME" container inspect "${mgmt_container}-lmcache" >/dev/null 2>&1; then
    printf 'stopping %s\n' "${mgmt_container}-lmcache"
    "$CONTAINER_RUNTIME" stop -t 10 "${mgmt_container}-lmcache" >/dev/null
    "$CONTAINER_RUNTIME" rm "${mgmt_container}-lmcache" >/dev/null
    printf 'removed %s\n' "${mgmt_container}-lmcache"
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
    # expect means a fallback: no "NVFP4-CSF layer" lines = experts not on the
    # CSF path; no "b12x weights loaded" with LOAD_FORMAT=b12x = another
    # loader; no RoCEnante line = NCCL for every collective. This image logs
    # no A4-prefill marker (the CSF layer line names the decode mode only);
    # --check counts the expert input scales A4 prefill needs instead.
    "$CONTAINER_RUNTIME" logs "$mgmt_container" 2>&1 | grep -E \
      'speculative_config|NVFP4-CSF layer 3 |modelopt_mixed|b12x weights loaded|[Ii]nstant[Tt]ensor|io_uring|Loading weights took|Model loading took|B12X fetched|Using .* all-reduce backends|RoCEnante|kda_prefill|split GLM-5.3 cache pages|physical page sizes|Rebalanced split KV|Initial free memory|GPU KV cache size|Maximum concurrency|draft-only|vocabulary head|Graph capturing finished|init engine|fastokens|Application startup complete' \
      | sed 's/^/  /'
    if [ "${NODE_RANK:-}" = 0 ]; then
      printf 'health: '; curl -fsS "http://127.0.0.1:${API_PORT:-8000}/health" 2>/dev/null && echo " OK" || echo " not ready"
    fi
    exit 0
    ;;
  --status)
    require_runtime
    "$CONTAINER_RUNTIME" ps -a --filter "name=^/${mgmt_container}(-lmcache)?$" \
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
: "${SERVING_IMAGE:=local/vllm:karmic-kraken-intcache1008-cu132}"
: "${SHM_SIZE:=16g}"
: "${FABRIC_PROFILE:=single}"
: "${B12X_ROCE_HCA:=}"
: "${VLLM_USE_FASTOKENS:=0}"
: "${CHECKPOINT_IDENTITY_SHA256:=}"
: "${CHECKPOINT_REVISION:=}"            # empty = unchecked; 7-40 hex = hf download commit prefix
: "${LOAD_FORMAT:=b12x}"                 # b12x | instanttensor | fastsafetensors | safetensors | auto
# InstantTensor staging bounds (LOAD_FORMAT=instanttensor only). BUFFER_SIZE
# 1.25 GiB covers the largest tensor (the 1,268,776,960 B BF16 vocab head).
: "${INSTANTTENSOR_BUFFER_SIZE:=1342177280}"
: "${INSTANTTENSOR_IO_DEPTH:=3}"
: "${INSTANTTENSOR_CONCURRENCY:=1}"
: "${INSTANTTENSOR_CHUNK_SIZE:=8388608}"
: "${INSTANTTENSOR_COPY:=auto}"
: "${MEM_PREFLIGHT:=die}"
: "${PREFIX_CACHE:=none}"                # none | sparkcache | lmcache (env section 9)
: "${PREFIX_CACHE_HOST_PATH:=}"
: "${PREFIX_CACHE_DISK_GB:=200}"
: "${PREFIX_CACHE_MODEL_IDENTITY:=auto}"
: "${PREFIX_CACHE_SPARK_MIN_SPAN_TOKENS:=4096}"
: "${PREFIX_CACHE_SPARK_MAX_SPAN_TOKENS:=262144}"
: "${PREFIX_CACHE_SPARK_ACCESS_MODE:=read-write}"
: "${PREFIX_CACHE_SPARK_LOAD_THREADS:=1}"
: "${PREFIX_CACHE_SPARK_SHARED_PREFIX_TTL:=}"
: "${PREFIX_CACHE_LMC_L1_GB:=2}"
: "${PREFIX_CACHE_LMC_PORT:=}"
: "${PREFIX_CACHE_LMC_RANK1_ADDR:=}"
: "${PREFIX_CACHE_LMC_L2_WORKERS:=4}"
: "${PREFIX_CACHE_LMC_CPU_WORKERS:=4}"

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
#   PREFILL_ACTIVATIONS a4 -> B12X_W4A16_A4_PREFILL=1: EVERY prefill row of
#     the W4A16 MoE calls runs with NVFP4 activations over the same packed
#     weights (short prompts, chunk tails and multi-turn deltas included);
#     decode, verification, graph replay and draft rows stay A16. a16 -> 0.
#     B12X engages A4 only for experts with calibrated input scales and
#     falls back to A16 silently otherwise: --check counts them.
# The checkpoint card asks for FORCE_A16=1 + FP32 top-k weights; the release
# measured a4 "as accurate as a16" on this QAD checkpoint and 12-14% faster
# prefill at TP4 (and refuses a4 only for the pre-QAD -Spark checkpoint).
case "$EXPERT_ACTIVATIONS" in bf16|fp4) : ;; *) die "EXPERT_ACTIVATIONS must be bf16 or fp4: $EXPERT_ACTIVATIONS" ;; esac
case "$ROUTER_WEIGHTS" in fp32|bf16) : ;; *) die "ROUTER_WEIGHTS must be fp32 or bf16: $ROUTER_WEIGHTS" ;; esac
case "$PREFILL_ACTIVATIONS" in a4|a16) : ;; *) die "PREFILL_ACTIVATIONS must be a4 or a16: $PREFILL_ACTIVATIONS" ;; esac
if [ "$EXPERT_ACTIVATIONS" = fp4 ]; then
  [ "$ROUTER_WEIGHTS" = bf16 ] \
    || die "ROUTER_WEIGHTS=fp32 applies to W4A16 experts only; with EXPERT_ACTIVATIONS=fp4 set ROUTER_WEIGHTS=bf16"
  [ "$PREFILL_ACTIVATIONS" = a16 ] \
    || die "PREFILL_ACTIVATIONS=a4 needs EXPERT_ACTIVATIONS=bf16 (with fp4 every row is already W4A4); set PREFILL_ACTIVATIONS=a16"
  warn "EXPERT_ACTIVATIONS=fp4 (W4A4 decode) is off the checkpoint card's accuracy settings (W4A16 + FP32 router weights); quality A/B only"
fi
moe_force_a16=0; [ "$EXPERT_ACTIVATIONS" = bf16 ] && moe_force_a16=1
fp32_topk=0; [ "$ROUTER_WEIGHTS" = fp32 ] && fp32_topk=1
precision_env_args=(
  -e "VLLM_B12X_MOE_FP4_FORCE_A16=$moe_force_a16"
  -e "B12X_W4A16_FP32_TOPK_WEIGHTS=$fp32_topk"
)
# Release launcher: GLM without a speculator quantizes the NVFP4 activations
# of the dense path ("GLM non-speculative activation policy").
[ "$SPECULATOR" != none ] || precision_env_args+=(-e VLLM_B12X_NVFP4_ACTIVATION_MODE=quantized)
a4_switch=0; [ "$PREFILL_ACTIVATIONS" = a4 ] && a4_switch=1
precision_env_args+=(-e "B12X_W4A16_A4_PREFILL=$a4_switch")
precision_desc="experts $EXPERT_ACTIVATIONS, router $ROUTER_WEIGHTS, prefill $PREFILL_ACTIVATIONS$([ "$a4_switch" = 0 ] || printf ' (every prefill row)')"

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

# --------------------------------------------------- image capabilities
# Source probe, not imports: the loader registries pull in modules that need a
# GPU driver at import time. Required of the image:
#   reader   recipes: ModelOpt recipes declare the CSF expert scales
#            (weight_scale_encoding: csf; vLLM >= 259c4228). An image with the
#            older dedicated nvfp4_csf loader is refused.
#   b        b12x CSF expert weights (Nvfp4CsfWeights, CsfScalePlanes)
#   a4       b12x W4A16 A4 prefill kernels, switched by B12X_W4A16_A4_PREFILL
# Reported:
#   ds       NVFP4 MTP draft head accepted on all of SM12x (vLLM #995)
#   where    flashinfer (flashinfer/experimental/b12x) | standalone b12x
#   bl       native --load-format b12x: vLLM's B12xModelLoader (d0f0cad9,
#            registered on first use, no plugin) + b12x.loader's C sources
#            (FlashInfer package data; built at first load against liburing)
# Without the image (absent locally, or SKIP_IMAGE_CHECKS=1) the launcher
# assumes an image that has all of them, and says so.
image_probed=0
if [ "${SKIP_IMAGE_CHECKS:-0}" != 1 ] \
   && command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1 \
   && "$CONTAINER_RUNTIME" image inspect "$SERVING_IMAGE" >/dev/null 2>&1; then
  csf_probe=$("$CONTAINER_RUNTIME" run --rm -i --entrypoint /opt/venv/bin/python "$SERVING_IMAGE" - <<'PY' 2>/dev/null || true
import importlib.metadata as md, importlib.util, pathlib
vroot = pathlib.Path(importlib.util.find_spec("vllm").origin).parent
def vsrc(rel):
    path = vroot / rel
    return path.read_text(errors="ignore") if path.is_file() else ""
loaders = vsrc("model_executor/model_loader/__init__.py")
modelopt = vsrc("model_executor/layers/quantization/modelopt.py")
if '"nvfp4_csf"' in loaders:
    reader = "nvfp4_csf"
elif "weight_scale_encoding" in modelopt:
    reader = "recipes"
else:
    reader = "none"
fi = importlib.util.find_spec("flashinfer")
root = pathlib.Path(fi.origin).parent / "experimental" / "b12x" if fi and fi.origin else None
if root is not None and root.is_dir():
    where = "flashinfer"
else:
    root, where = pathlib.Path(importlib.util.find_spec("b12x").origin).parent, "standalone"
b12x_src = "".join(f.read_text(errors="ignore") for f in (root / "moe").rglob("*.py"))
b = all(f"class {n}" in b12x_src for n in ("Nvfp4CsfWeights", "CsfScalePlanes"))
a4 = ((root / "moe" / "_shared" / "kernels" / "w4a16" / "prefill_a4.py").is_file()
      and '"B12X_W4A16_A4_PREFILL"' in vsrc("utils/b12x.py"))
ds = "if major != 12" in vsrc("models/glm5next/nvidia/mtp_draft_head.py")
bl = ('load_format == "b12x"' in loaders and "B12xModelLoader" in loaders
      and bool(vsrc("model_executor/model_loader/b12x_loader.py"))
      and (root / "loader" / "_bounce.c").is_file())
print("CSF", reader, int(b), int(a4), md.version("nvidia-cutlass-dsl"), int(ds), where, int(bl))
PY
)
  case "$csf_probe" in
    "CSF recipes 1 1 "*) : ;;
    "CSF nvfp4_csf "*) die "$SERVING_IMAGE reads CSF with the older dedicated nvfp4_csf loader; this launcher serves the ModelOpt-recipe CSF reader only (local/vllm:karmic-kraken-intcache1008-cu132, dgx-spark-builder/build-kk-integration-cache-cu132.env)" ;;
    "CSF "*) die "$SERVING_IMAGE cannot serve this checkpoint (probe: reader, b12x CSF weights, A4 prefill = ${csf_probe#CSF }). Build dgx-spark-builder/build-kk-integration-cache-cu132.env" ;;
    *) die "image capability probe failed to run in $SERVING_IMAGE (output: ${csf_probe:-none}); SKIP_IMAGE_CHECKS=1 to bypass" ;;
  esac
  read -r _ _ _ _ csf_dsl csf_ds csf_b12x_where csf_b12x_loader <<< "$csf_probe"
  image_probed=1
else
  csf_dsl=unprobed; csf_ds=1; csf_b12x_where=unprobed; csf_b12x_loader=1
  warn "$SERVING_IMAGE not probed (absent locally or SKIP_IMAGE_CHECKS=1): assuming the ModelOpt-recipe CSF reader, the native b12x loader and A4 prefill kernels"
fi
case "$LOAD_FORMAT" in
  b12x|instanttensor|fastsafetensors|safetensors|auto) : ;;
  *) die "LOAD_FORMAT must be b12x, instanttensor, fastsafetensors, safetensors, or auto: $LOAD_FORMAT" ;;
esac
case "$INSTANTTENSOR_COPY" in
  auto) : ;;
  *) die "INSTANTTENSOR_COPY=$INSTANTTENSOR_COPY: the instanttensor_copy loader option was removed in vLLM 6575b5ac (2026-09-05); leave it auto" ;;
esac
[ "$LOAD_FORMAT" != b12x ] || [ "${csf_b12x_loader:-0}" = 1 ] \
  || die "LOAD_FORMAT=b12x but $SERVING_IMAGE has no native b12x load format (vLLM B12xModelLoader, d0f0cad9) or its b12x ships no loader sources (b12x/loader/_bounce.c). Set LOAD_FORMAT=instanttensor (and uncomment its block in env section 7), or rebuild"
case ",${VLLM_PLUGINS-}," in
  *,b12x_loader,*) warn "VLLM_PLUGINS names b12x_loader: this vLLM has the b12x loader built in and FlashInfer's b12x registers no plugin entry point, so the entry does nothing. Set VLLM_PLUGINS= (empty)" ;;
esac

# ------------------------------------------------- FP4-CSF checkpoint
# MODEL_HOST_PATH is the hf download root of the Hugging Face-layout
# checkpoint (HF main from dec48abd): config.json (ModelOpt recipes with
# weight_scale_encoding: csf) + model.safetensors.index.json + the indexed
# shards. vLLM reads the directory itself. Validated: config recipes,
# Glm5Next architecture, CSF streams in the index, every indexed shard present
# and non-empty. Identity = LIL's csf_hf_content_digest (runtime/launcher.py):
# config.json, the index, and per shard its LFS SHA-256 -- read from the hf
# download metadata (<root>/.cache/huggingface/download/) or the Hub-cache
# blob name, never by hashing the ~178 GB of shards; a shard with neither
# counts by size (and can then not match LIL's pinned digest). The revision
# comes from the same metadata. Shards must resolve inside MODEL_HOST_PATH (a
# Hub-cache snapshot links into ../../blobs, which the mount cannot see):
# download with --local-dir. Compare the printed identity between ranks, or
# pin it with CHECKPOINT_IDENTITY_SHA256 in both env files.
command -v python3 >/dev/null 2>&1 \
  || die "python3 is required on the host to validate the CSF checkpoint"
model_container_path=/models/glm-5.3-flash-csf
csf_out=$(python3 - "$MODEL_HOST_PATH" <<'PYEOF'
import hashlib, json, os, pathlib, re, sys

root = pathlib.Path(sys.argv[1])
HEX64 = re.compile(r"[0-9a-f]{64}")
CSF_STREAMS = (".nvfp4_csf_fixed", ".nvfp4_csf_exceptions")
# blackwell-llm-docker runtime/checkpoints.yaml "contents" (docker-140):
# Hugging Face-layout revision -> content identity.
LIL_HF_CONTENTS = {
    "dec48abd33efa73c3bb7c95b74eee10cad34f9be":
        "bc453166d2d32020847a51d8aedab725bf96c9e6e25635181c8a0ca49530ec0e",
}

def fail(msg):
    print("ERROR " + msg)
    sys.exit(0)

def quant_config(config, where):
    holder = config if "quantization_config" in config else config.get("text_config")
    if not isinstance(holder, dict) or not isinstance(holder.get("quantization_config"), dict):
        fail(f"{where} has no quantization_config")
    return holder

def recipes(quant):
    layers = quant.get("quantized_layers")
    if layers is None:
        layers = (quant.get("quantization") or {}).get("quantized_layers")
    return layers if isinstance(layers, dict) else {}

def vision_stored(quant):
    return int(any(n.startswith(("model.visual.", "visual.")) for n in recipes(quant)))

def hf_metadata(name):
    """(commit, etag) that hf download --local-dir recorded for a file."""
    meta = root / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    try:
        lines = meta.read_text().splitlines()
    except OSError:
        return None, None
    commit = lines[0].strip() if lines else None
    etag = lines[1].strip().strip('"') if len(lines) > 1 else None
    return commit or None, etag or None

def hf_layout():
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    try:
        config_raw = config_path.read_bytes()
        index_raw = index_path.read_bytes()
        config = json.loads(config_raw)
        index = json.loads(index_raw)
    except ValueError as error:
        fail(f"config.json or model.safetensors.index.json is not JSON: {error}")
    arch = config.get("architectures") or []
    if not any(a.startswith("Glm5Next") for a in arch):
        fail(f"config.json architectures {arch}; expected Glm5Next* (GLM-5.3-Flash)")
    quant = quant_config(config, "config.json")["quantization_config"]
    if not any(isinstance(r, dict) and r.get("weight_scale_encoding") == "csf"
               for r in recipes(quant).values()):
        fail("config.json has no ModelOpt recipe with weight_scale_encoding: csf -- not an FP4-CSF checkpoint")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        fail("model.safetensors.index.json has no weight_map")
    if not any(name.endswith(CSF_STREAMS) for name in weight_map):
        fail("model.safetensors.index.json names no .nvfp4_csf_fixed/.nvfp4_csf_exceptions streams -- not an FP4-CSF checkpoint")
    shards = sorted(set(weight_map.values()))
    # Calibrated routed-expert activation scales: B12X runs A4 prefill only
    # for experts that have them (dec48abd: 36,288).
    input_scales = sum(1 for n in weight_map if ".experts." in n and n.endswith(".input_scale"))

    revisions = set()
    for name in ("config.json", "model.safetensors.index.json"):
        commit, _ = hf_metadata(name)
        if commit:
            revisions.add(commit)
    # Same byte stream as runtime/launcher.py csf_hf_content_digest(); a
    # local-dir download records the LFS SHA-256 as the file's etag, which is
    # what the Hub cache names the blob, so both download modes agree.
    digest = hashlib.sha256(b"lil-fp4-csf-hf-v1\0")
    digest.update(config_raw + b"\0")
    digest.update(index_raw + b"\0")
    missing, total, by_size = [], 0, 0
    real_root = os.path.realpath(root)
    for name in shards:
        rel = pathlib.PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".safetensors":
            fail(f"the index names a shard outside the checkpoint: {name}")
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        target = pathlib.Path(os.path.realpath(path))
        if os.path.commonpath([real_root, str(target)]) != real_root:
            # A Hub-cache snapshot links into ../../blobs: the read-only
            # mount of MODEL_HOST_PATH cannot follow it in the container.
            fail(f"{name} resolves outside MODEL_HOST_PATH ({target}); a Hub-cache "
                 "snapshot does not survive the container mount -- download with "
                 "'hf download ... --local-dir <dir>' and point MODEL_HOST_PATH there")
        size = target.stat().st_size
        if size == 0:
            missing.append(f"{name} (empty)")
            continue
        total += size
        commit, etag = hf_metadata(name)
        if commit:
            revisions.add(commit)
        if target.parent.name == "blobs" and HEX64.fullmatch(target.name):
            content = target.name
        elif etag and HEX64.fullmatch(etag):
            content = etag
        else:
            content = str(size)
            by_size += 1
        digest.update(f"{name}\0{content}\0".encode())
    if missing:
        more = f" and {len(missing) - 6} more" if len(missing) > 6 else ""
        fail(f"incomplete download ({len(missing)} of {len(shards)} indexed shards missing or empty): " + ", ".join(missing[:6]) + more)
    identity = digest.hexdigest()
    if len(revisions) == 1:
        revision = next(iter(revisions))
    else:
        revision = "mixed" if revisions else "-"
    if by_size:
        lil = "size-based"
    elif revision in LIL_HF_CONTENTS:
        lil = "match" if LIL_HF_CONTENTS[revision] == identity else "mismatch"
    else:
        lil = "match" if identity in LIL_HF_CONTENTS.values() else "unpinned"
    # identity shards bytes revision lil vision by_size input_scales
    print(f"OK {identity} {len(shards)} {total} {revision} {lil} {vision_stored(quant)} {by_size} {input_scales}")

if (root / "manifest.json").is_file():
    fail("manifest.json: a container-layout checkpoint (HF revisions up to fd660d51). This launcher "
         "serves the Hugging Face layout only: download HF main (dec48abd or later) with "
         "'hf download --local-dir' and point MODEL_HOST_PATH there")
elif (root / "config.json").is_file() and (root / "model.safetensors.index.json").is_file():
    hf_layout()
else:
    fail(f"{root} has no config.json + model.safetensors.index.json. Point MODEL_HOST_PATH "
         "at the hf download root of GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD")
PYEOF
) || die "CSF checkpoint validation failed to run (python3 error above)"
case "$csf_out" in
  OK\ *) read -r _ csf_identity csf_shards csf_bytes csf_revision csf_lil csf_vision_stored csf_by_size csf_input_scales <<< "$csf_out" ;;
  ERROR\ *) die "MODEL_HOST_PATH: ${csf_out#ERROR }" ;;
  *) die "CSF checkpoint validation produced no result: $csf_out" ;;
esac
case "$csf_lil" in
  mismatch) warn "MODEL_HOST_PATH: revision ${csf_revision:0:12} but content identity $csf_identity differs from the one blackwell-llm-docker pins for it -- config.json, the index or a shard is not the published file (edited, or re-downloaded over another revision?)" ;;
  size-based) warn "MODEL_HOST_PATH: $csf_by_size shard(s) carry no hf download metadata (.cache/huggingface/download/), so the identity counts them by size: still valid between ranks copied the same way, but it cannot match LIL's pinned digest. Download with 'hf download --local-dir' on each node to get the full identity" ;;
esac
[ "$csf_revision" != mixed ] \
  || warn "MODEL_HOST_PATH: the hf download metadata names more than one revision; files from two revisions may be mixed. Re-download into an empty directory"
[ "$PREFILL_ACTIVATIONS" != a4 ] || [ "$csf_input_scales" -gt 0 ] \
  || warn "PREFILL_ACTIVATIONS=a4 but the checkpoint index names no routed-expert input_scale tensors: B12X needs calibrated input scales for A4 prefill and silently runs A16 without them"
if [ -n "$CHECKPOINT_IDENTITY_SHA256" ]; then
  printf '%s' "$CHECKPOINT_IDENTITY_SHA256" | grep -Eq '^[0-9a-f]{64}$' \
    || die "CHECKPOINT_IDENTITY_SHA256 must be 64 lowercase hex chars (copy the identity --check prints): $CHECKPOINT_IDENTITY_SHA256"
  [ "$CHECKPOINT_IDENTITY_SHA256" = "$csf_identity" ] \
    || die "checkpoint identity $csf_identity != CHECKPOINT_IDENTITY_SHA256 $CHECKPOINT_IDENTITY_SHA256 -- the two nodes do not hold the same checkpoint (or MODEL_HOST_PATH moved to another revision: re-pin both files)"
fi
# CHECKPOINT_REVISION: the Hub commit (full or >= 7-hex prefix) MODEL_HOST_PATH
# must have been downloaded at, read from the hf --local-dir metadata. Guards
# against pointing a rank file at the wrong per-revision folder; the content
# itself is pinned by CHECKPOINT_IDENTITY_SHA256.
if [ -n "$CHECKPOINT_REVISION" ]; then
  printf '%s' "$CHECKPOINT_REVISION" | grep -Eq '^[0-9a-f]{7,40}$' \
    || die "CHECKPOINT_REVISION must be 7-40 lowercase hex chars (a Hub commit or its prefix): $CHECKPOINT_REVISION"
  case "$csf_revision" in
    -) die "CHECKPOINT_REVISION=$CHECKPOINT_REVISION but $MODEL_HOST_PATH carries no hf download metadata (.cache/huggingface/download/) to read a revision from; download with 'hf download --revision <sha> --local-dir <dir>' or leave CHECKPOINT_REVISION empty" ;;
    mixed) die "CHECKPOINT_REVISION=$CHECKPOINT_REVISION but the hf download metadata in $MODEL_HOST_PATH names more than one revision; re-download into an empty directory" ;;
    "$CHECKPOINT_REVISION"*) : ;;
    *) die "MODEL_HOST_PATH was downloaded at revision ${csf_revision:0:12}, not CHECKPOINT_REVISION=$CHECKPOINT_REVISION -- wrong per-revision folder?" ;;
  esac
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

# Vocabulary heads (GB10 is capability 12.1). Three variables decide which
# weight produces MTP draft logits (vLLM 19f2c20e):
#   VLLM_MTP_NVFP4_LM_HEAD=1 (vLLM default 1): the MTP layer loads its own
#     runtime-quantized NVFP4 head ("path A").
#   VLLM_MXFP8_LM_HEAD=1: the target head is quantized to MXFP8 at load; with
#     MTP_NVFP4_LM_HEAD=0 the drafter shares it.
#   VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4 ("path B"): an NVFP4 copy of the target's
#     BF16 head on FlashInfer's CuTe-DSL W4A16 GEMM, accepted on all of SM12x
#     since vLLM #995. glm5next/nvidia/mtp.py
#     prepare_draft_lm_head() skips the copy WITHOUT a log line when the head
#     it receives is runtime-quantized, so nvfp4 only takes effect with BOTH
#     VLLM_MTP_NVFP4_LM_HEAD=0 and VLLM_MXFP8_LM_HEAD=0 (BF16 verifier head:
#     the MXFP8 head's measured +2-4% steps/s at C1 is given up).
# VLLM_GLM53_L2_PREFETCH is off by default on 12.1; =1 forces it. The GLM
#   windows (20/50/15 MB) are sized for the 128 MB L2 of an RTX PRO 6000;
#   GB10 has 24 MB (upstream's GB10 DS4.1 prefetcher runs 20/16/20 MB with
#   a 4-CTA grid, which this pin already selects on 12.1).
draft_head=${VLLM_GLM53_MTP_DRAFT_HEAD:-bf16}
case "$draft_head" in
  bf16) : ;;
  nvfp4)
    [ "$SPECULATOR" = mtp ] \
      || warn "VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4 only affects SPECULATOR=mtp; it does nothing with SPECULATOR=$SPECULATOR"
    [ "${VLLM_MTP_NVFP4_LM_HEAD:-1}" = 0 ] \
      || die "VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4 with VLLM_MTP_NVFP4_LM_HEAD=${VLLM_MTP_NVFP4_LM_HEAD:-1 (vLLM default)}: the drafter keeps its own NVFP4 head and vLLM silently skips the draft-head copy. Set VLLM_MTP_NVFP4_LM_HEAD=0 (and VLLM_MXFP8_LM_HEAD=0)"
    [ "${VLLM_MXFP8_LM_HEAD:-0}" = 0 ] \
      || die "VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4 with VLLM_MXFP8_LM_HEAD=$VLLM_MXFP8_LM_HEAD: the drafter shares the MXFP8 target head and vLLM silently skips the draft-head copy. Set VLLM_MXFP8_LM_HEAD=0 (the verifier head becomes BF16)"
    ;;
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

# Image checks (only when the image is present locally): LD_PRELOAD paths
# and fastokens. CSF reader, b12x, A4 prefill and draft head were probed in
# the image capabilities section.
if [ "$image_probed" = 1 ]; then
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

  [ "$draft_head" != nvfp4 ] || [ "${csf_ds:-0}" = 1 ] \
    || die "VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4 but $SERVING_IMAGE's vLLM still gates the NVFP4 draft head to capability 12.0 (GB10 is 12.1; lifted in vLLM #995). Use local/vllm:karmic-kraken-intcache1008-cu132 (build-kk-integration-cache-cu132.env) or set bf16"
  image_desc="recipe CSF reader; b12x $csf_b12x_where$([ "${csf_b12x_loader:-0}" = 1 ] && printf ' (native loader)' || printf ' (no native loader)'); A4 prefill kernels; SM12x draft head $([ "${csf_ds:-0}" = 1 ] && echo yes || echo no); CuTe DSL $csf_dsl"

  if [ "$VLLM_USE_FASTOKENS" = 1 ]; then
    fk_label=$("$CONTAINER_RUNTIME" image inspect --format '{{index .Config.Labels "org.local-inference.fastokens.version"}}' "$SERVING_IMAGE" 2>/dev/null || true)
    [ -n "$fk_label" ] && [ "$fk_label" != "<no value>" ] \
      || die "VLLM_USE_FASTOKENS=1 but $SERVING_IMAGE carries no fastokens (no org.local-inference.fastokens.version label); vLLM would stop at tokenizer load. Rebuild with PATCH_FASTOKENS=on or set 0"
    image_desc="$image_desc; fastokens $fk_label"
  fi
else
  image_desc="not probed (assumed: recipe CSF reader, native b12x loader, A4 prefill kernels)"
fi

# ----------------------------------------------------------- launch command

container_name="$mgmt_container"
dflash_container_path=/models/glm-5.3-flash-dflash2
chat_template_container_path=/models/chat_template.jinja

# The checkpoint's own chat_template.jinja (dec48abd) renders assistant
# content as content.strip(). A returned answer then re-renders to tokens
# that differ from the generated ones, and the next turn can only reuse the
# turn-1 prompt, not the response checkpoint: the reply is prefilled again.
# glm-53-flash-csf_chat-template.jinja is LIL's runtime template
# (runtime/templates/glm53-flash.jinja, sha256 b5bbea09...): the official one
# with assistant content rendered exactly as generated. Relative paths are
# resolved against this script's folder.
if [ -n "$CHAT_TEMPLATE_HOST_PATH" ]; then
  case "$CHAT_TEMPLATE_HOST_PATH" in /*) ;; *) CHAT_TEMPLATE_HOST_PATH="$script_dir/$CHAT_TEMPLATE_HOST_PATH" ;; esac
  [ -f "$CHAT_TEMPLATE_HOST_PATH" ] || die "CHAT_TEMPLATE_HOST_PATH does not exist: $CHAT_TEMPLATE_HOST_PATH"
fi

# The INSTANTTENSOR_* staging bounds are read by the instanttensor library
# only; under any other loader they are inert, so they are passed only when
# that loader is selected (the env file's own copies still reach the
# container via --env-file, which is harmless).
instanttensor_env_args=()
if [ "$LOAD_FORMAT" = instanttensor ]; then
  instanttensor_env_args=(
    -e INSTANTTENSOR_BUFFER_SIZE="$INSTANTTENSOR_BUFFER_SIZE"
    -e INSTANTTENSOR_IO_DEPTH="$INSTANTTENSOR_IO_DEPTH"
    -e INSTANTTENSOR_CONCURRENCY="$INSTANTTENSOR_CONCURRENCY"
    -e INSTANTTENSOR_CHUNK_SIZE="$INSTANTTENSOR_CHUNK_SIZE"
  )
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
      # follows the served directory and load format.
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
      # draft_load_config it inherits the target's LOAD_FORMAT. The release
      # profile pins load_format auto for exactly this.
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

# ===== BEGIN prefix cache (PREFIX_CACHE: none | sparkcache | lmcache) =====
# Ported from glm-5.3-flash-cache_pair-serve.sh. Builds --kv-transfer-config
# and the extra serve flags, mounts the per-node NVMe directory, and
# (lmcache) describes the CPU-only lmcache server container started before
# vLLM in --run. See env section 9. Cache identity = the checkpoint identity,
# DCP_SIZE in the namespace, REPLAYSSM forced to 0.
prefix_cache_desc="none (GPU prefix cache only, ${RECURRENT_CHECKPOINT_POLICY:-auto})"
lmc_server_cmd=()
lmc_container="${mgmt_container}-lmcache"
lmc_health_url=""
case "$PREFIX_CACHE" in
  none) ;;
  sparkcache|lmcache)
    [ -n "$PREFIX_CACHE_HOST_PATH" ] || die "PREFIX_CACHE=$PREFIX_CACHE needs PREFIX_CACHE_HOST_PATH (a directory on this node's NVMe)"
    case "$PREFIX_CACHE_HOST_PATH" in /*) ;; *) die "PREFIX_CACHE_HOST_PATH must be absolute: $PREFIX_CACHE_HOST_PATH" ;; esac
    PREFIX_CACHE_HOST_PATH=${PREFIX_CACHE_HOST_PATH%/}
    case "$PREFIX_CACHE_HOST_PATH/" in
      "${CACHE_HOST_PATH%/}"/*) die "PREFIX_CACHE_HOST_PATH must not be inside CACHE_HOST_PATH (--clear wipes the JIT cache): $PREFIX_CACHE_HOST_PATH" ;;
      "${MODEL_HOST_PATH%/}"/*) die "PREFIX_CACHE_HOST_PATH must not be inside MODEL_HOST_PATH: $PREFIX_CACHE_HOST_PATH" ;;
    esac
    mkdir -p "$PREFIX_CACHE_HOST_PATH" 2>/dev/null || true
    [ -d "$PREFIX_CACHE_HOST_PATH" ] && [ -w "$PREFIX_CACHE_HOST_PATH" ] || die "PREFIX_CACHE_HOST_PATH is not a writable directory: $PREFIX_CACHE_HOST_PATH"
    require_positive_integer PREFIX_CACHE_DISK_GB
    [ "$ENABLE_PREFIX_CACHING" = 1 ] || die "PREFIX_CACHE=$PREFIX_CACHE needs ENABLE_PREFIX_CACHING=1"

    # GLM KDA speculative recovery requires an atomic request-boundary
    # connector (vLLM: "GLM KDA recovery requires an atomic request-boundary
    # external-cache connector"); neither connector here is one.
    [ "$REPLAYSSM" != 1 ] || die "REPLAYSSM=1 with PREFIX_CACHE=$PREFIX_CACHE: GLM KDA recovery needs a request-boundary connector; set REPLAYSSM= (the launcher then passes --no-use-replayssm) or 0"
    [ "$REPLAYSSM" = 0 ] || extra_args+=(--no-use-replayssm)
    [ "$DCP_SIZE" = 1 ] \
      || warn "PREFIX_CACHE=$PREFIX_CACHE with DCP_SIZE=$DCP_SIZE: no connector has been run with decode context parallelism on this pair (the release qualifies TP2 external cache at DCP1 only)"

    # Cache identity of the target: explicit 64-hex, or auto = the checkpoint
    # identity printed above (LIL's content digest over config.json, the index
    # and every shard's LFS SHA-256). Another checkpoint revision is another
    # namespace: caches written for the old one are not read.
    if [ "$PREFIX_CACHE_MODEL_IDENTITY" = auto ]; then
      pc_target=$csf_identity
    else
      printf '%s' "$PREFIX_CACHE_MODEL_IDENTITY" | grep -Eq '^[0-9a-f]{64}$' \
        || die "PREFIX_CACHE_MODEL_IDENTITY must be auto or 64 lowercase hex: $PREFIX_CACHE_MODEL_IDENTITY"
      pc_target=$PREFIX_CACHE_MODEL_IDENTITY
    fi
    # Serving layout: anything that changes the bytes of a stored page or
    # the meaning of a boundary gets its own namespace directory.
    pc_layout=$(printf '%s|' "$PREFIX_CACHE" tp2 "dcp$DCP_SIZE" "$BLOCK_SIZE" "$KV_CACHE_DTYPE" \
      "${VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE:-coupled}" "${VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE:-coupled}" \
      "$RECURRENT_CHECKPOINT_POLICY" "$SPECULATOR" "$NUM_SPECULATIVE_TOKENS" "$MAX_NUM_BATCHED_TOKENS" \
      | sha256sum | cut -c1-12)
    pc_namespace="/prefix-cache/$PREFIX_CACHE/${pc_target:0:16}-$pc_layout"
    mount_args+=(-v "$PREFIX_CACHE_HOST_PATH:/prefix-cache")
    pc_disk_bytes=$(( PREFIX_CACHE_DISK_GB * 1024 * 1024 * 1024 ))

    # Image capability check (only when the image is present locally).
    pc_image_label() { "$CONTAINER_RUNTIME" image inspect --format "{{index .Config.Labels \"$1\"}}" "$SERVING_IMAGE" 2>/dev/null || true; }
    pc_have_image=0
    if [ "${SKIP_IMAGE_CHECKS:-0}" != 1 ] && command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1 \
       && "$CONTAINER_RUNTIME" image inspect "$SERVING_IMAGE" >/dev/null 2>&1; then
      pc_have_image=1
    fi
    ;;
  *) die "PREFIX_CACHE must be none, sparkcache, or lmcache: $PREFIX_CACHE" ;;
esac

if [ "$PREFIX_CACHE" = sparkcache ]; then
  if [ "$pc_have_image" = 1 ]; then
    pc_sc_patches=$(pc_image_label org.local-inference.sparkcache.vllm-patches)
    case "$pc_sc_patches" in
      *vmm-exemption*) : ;;
      *) die "PREFIX_CACHE=sparkcache: $SERVING_IMAGE carries no SparkCache vmm-exemption patch (label: '${pc_sc_patches:-none}'); vLLM would refuse the connector under expandable_segments. Rebuild with PATCH_SPARKCACHE=on" ;;
    esac
    if [ -n "$PREFIX_CACHE_SPARK_SHARED_PREFIX_TTL" ]; then
      case "$pc_sc_patches" in *shared-prefix*) : ;; *) warn "PREFIX_CACHE_SPARK_SHARED_PREFIX_TTL is set but the image has no shared-prefix-lease patch; the setting has no effect" ;; esac
    fi
  fi
  [ "$RECURRENT_CHECKPOINT_POLICY" = aligned ] \
    || warn "PREFIX_CACHE=sparkcache with RECURRENT_CHECKPOINT_POLICY=${RECURRENT_CHECKPOINT_POLICY:-auto}: SparkCache stores 256-token boundaries and needs a KDA state at each; with request boundaries almost nothing is storable. Set aligned"
  for name in PREFIX_CACHE_SPARK_MIN_SPAN_TOKENS PREFIX_CACHE_SPARK_MAX_SPAN_TOKENS; do
    require_positive_integer "$name"
    [ $(( ${!name} % 256 )) -eq 0 ] || die "$name must be a multiple of 256: ${!name}"
  done
  [ "$PREFIX_CACHE_SPARK_MAX_SPAN_TOKENS" -ge "$PREFIX_CACHE_SPARK_MIN_SPAN_TOKENS" ] \
    || die "PREFIX_CACHE_SPARK_MAX_SPAN_TOKENS must be >= PREFIX_CACHE_SPARK_MIN_SPAN_TOKENS"
  case "$PREFIX_CACHE_SPARK_ACCESS_MODE" in read-write|restore-only|store-only|disabled) : ;; *) die "PREFIX_CACHE_SPARK_ACCESS_MODE must be read-write, restore-only, store-only, or disabled: $PREFIX_CACHE_SPARK_ACCESS_MODE" ;; esac
  case "$PREFIX_CACHE_SPARK_LOAD_THREADS" in [1-8]) : ;; *) die "PREFIX_CACHE_SPARK_LOAD_THREADS must be 1-8: $PREFIX_CACHE_SPARK_LOAD_THREADS" ;; esac
  if [ -n "$PREFIX_CACHE_SPARK_SHARED_PREFIX_TTL" ]; then
    awk -v v="$PREFIX_CACHE_SPARK_SHARED_PREFIX_TTL" 'BEGIN{ exit !(v ~ /^[0-9]*\.?[0-9]+$/ && v+0 >= 1 && v+0 <= 300) }' \
      || die "PREFIX_CACHE_SPARK_SHARED_PREFIX_TTL must be 1-300 seconds: $PREFIX_CACHE_SPARK_SHARED_PREFIX_TTL"
  fi
  # Draft identity (SparkCache recomputes draft state after a restore, but
  # binds entries to the draft so a speculator change is a clean miss).
  # Formulas follow sparkcache deploy/glm53_flash/profile.py.
  case "$SPECULATOR" in
    mtp)
      if [ "$ADAPTIVE_SPECULATIVE_TOKENS" = 1 ]; then
        pc_policy="adaptive:${ADAPTIVE_SPECULATIVE_TOKENS_INITIAL}:${ADAPTIVE_SPECULATIVE_TOKENS_WINDOW}"
      else
        pc_policy=static
      fi
      pc_draft=$(printf 'glm53-embedded-mtp-v1\0%s\0%s\0%s' "$pc_target" "$NUM_SPECULATIVE_TOKENS" "$pc_policy" | sha256sum | cut -d' ' -f1)
      pc_draft_json=$(printf ',"spark_cache_draft_policy":"separate","spark_cache_draft_checkpoint_sha256":"%s"' "$pc_draft") ;;
    dflash2)
      [ -n "$DFLASH_WEIGHTS_SHA256" ] || die "PREFIX_CACHE=sparkcache with SPECULATOR=dflash2 needs DFLASH_WEIGHTS_SHA256 (the draft's cache identity)"
      pc_draft_json=$(printf ',"spark_cache_draft_policy":"separate","spark_cache_draft_checkpoint_sha256":"%s"' "$DFLASH_WEIGHTS_SHA256") ;;
    *) pc_draft_json=',"spark_cache_draft_policy":"colocated_target"' ;;
  esac
  pc_low_bytes=$(( pc_disk_bytes / 10 * 9 ))
  pc_extra=$(printf '"spark_cache_root":"%s","spark_cache_model_profile":"glm53-flash-hybrid","spark_cache_target_checkpoint_sha256":"%s"%s,"spark_cache_access_mode":"%s","spark_cache_publication_schema":"snapshot-v1","spark_cache_scheduler_probe":"none","spark_cache_streaming_snapshots":false,"spark_cache_cuda_restore":false,"spark_cache_max_bytes":%s,"spark_cache_low_watermark_bytes":%s,"spark_cache_ttl_seconds":0,"spark_cache_min_span_tokens":%s,"spark_cache_max_span_tokens":%s,"spark_cache_load_threads":%s' \
    "$pc_namespace" "$pc_target" "$pc_draft_json" "$PREFIX_CACHE_SPARK_ACCESS_MODE" "$pc_disk_bytes" "$pc_low_bytes" \
    "$PREFIX_CACHE_SPARK_MIN_SPAN_TOKENS" "$PREFIX_CACHE_SPARK_MAX_SPAN_TOKENS" "$PREFIX_CACHE_SPARK_LOAD_THREADS")
  [ -z "$PREFIX_CACHE_SPARK_SHARED_PREFIX_TTL" ] || pc_extra="$pc_extra,\"spark_cache_shared_prefix_lease_ttl_seconds\":$PREFIX_CACHE_SPARK_SHARED_PREFIX_TTL"
  pc_kv_transfer=$(printf '{"kv_connector":"SparkContextCacheConnector","kv_connector_module_path":"sparkcache.spark_context_cache_connector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{%s}}' "$pc_extra")
  extra_args+=(--kv-transfer-config "$pc_kv_transfer")
  # Host-RAM transient: one staged snapshot per load lane, <= ~7.5 KB/token.
  pc_ram_gib=$(awk -v t="$PREFIX_CACHE_SPARK_MAX_SPAN_TOKENS" -v n="$PREFIX_CACHE_SPARK_LOAD_THREADS" 'BEGIN{printf "%.1f", t*7500*n/1073741824}')
  prefix_cache_desc="sparkcache -> $PREFIX_CACHE_HOST_PATH (${PREFIX_CACHE_DISK_GB} GiB, spans ${PREFIX_CACHE_SPARK_MIN_SPAN_TOKENS}-${PREFIX_CACHE_SPARK_MAX_SPAN_TOKENS} tokens, ${PREFIX_CACHE_SPARK_ACCESS_MODE}; RAM transient <= ~${pc_ram_gib} GiB/node, take it off the KV pin)"
fi

if [ "$PREFIX_CACHE" = lmcache ]; then
  if [ "$pc_have_image" = 1 ] && [ -z "$(pc_image_label local-inference.lmcache.commit)" ]; then
    warn "PREFIX_CACHE=lmcache: $SERVING_IMAGE has no local-inference.lmcache.commit label; multi-server LMCacheMPConnector needs the integration/local-inference-lab LMCache (PATCH_LMCACHE_INTEGRATION)"
  fi
  # GLM's request-boundary bundles need LMCacheRecurrentCheckpointConnector,
  # which publishes a bundle only when every rank acks the SAME server --
  # impossible with one server per node. Aligned chunks carry their own KDA
  # state and the multi-server MP connector handles them.
  [ "$RECURRENT_CHECKPOINT_POLICY" = aligned ] \
    || die "PREFIX_CACHE=lmcache requires RECURRENT_CHECKPOINT_POLICY=aligned on a two-node pair (request-boundary bundles cannot span two lmcache servers)"
  require_positive_integer PREFIX_CACHE_LMC_L1_GB
  require_positive_integer PREFIX_CACHE_LMC_L2_WORKERS
  require_positive_integer PREFIX_CACHE_LMC_CPU_WORKERS
  [ -n "$PREFIX_CACHE_LMC_RANK1_ADDR" ] || die "PREFIX_CACHE=lmcache needs PREFIX_CACHE_LMC_RANK1_ADDR (rank 1's CX7 IPv4, identical in both files)"
  [ "$NODE_RANK" != 1 ] || [ "$PREFIX_CACHE_LMC_RANK1_ADDR" = "$VLLM_HOST_IP" ] \
    || die "PREFIX_CACHE_LMC_RANK1_ADDR ($PREFIX_CACHE_LMC_RANK1_ADDR) must equal rank 1's VLLM_HOST_IP ($VLLM_HOST_IP)"
  [ "$PREFIX_CACHE_LMC_RANK1_ADDR" != "$MASTER_ADDR" ] || die "PREFIX_CACHE_LMC_RANK1_ADDR must be rank 1's address, not MASTER_ADDR"
  pc_port=${PREFIX_CACHE_LMC_PORT:-$(( API_PORT + 10000 ))}
  require_port pc_port
  [ $(( pc_port + 2 )) -le 65535 ] || die "PREFIX_CACHE_LMC_PORT + 2 exceeds 65535: $pc_port"
  for p in "$pc_port" $(( pc_port + 1 )) $(( pc_port + 2 )); do
    [ "$p" != "$API_PORT" ] && [ "$p" != "$MASTER_PORT" ] || die "lmcache ports ($pc_port..$(( pc_port + 2 ))) collide with API_PORT/MASTER_PORT"
  done
  # LIL's GLM contract: a cache object is exactly one scheduler budget, and
  # must hold whole target pages.
  pc_chunk=$MAX_NUM_BATCHED_TOKENS
  if [ -n "${VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE:-}" ] && [ "$VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE" != auto ]; then
    [ $(( pc_chunk % VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE )) -eq 0 ] \
      || die "PREFIX_CACHE=lmcache: MAX_NUM_BATCHED_TOKENS ($pc_chunk) must be a multiple of VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE ($VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE)"
  fi
  if [ -d /dev/shm ]; then
    pc_shm_mb=$(df -Pm /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
    case "$pc_shm_mb" in ''|*[!0-9]*) ;; *) [ "$pc_shm_mb" -ge $(( PREFIX_CACHE_LMC_L1_GB * 1024 + 1024 )) ] || die "/dev/shm is ${pc_shm_mb} MiB; lmcache L1 (${PREFIX_CACHE_LMC_L1_GB} GiB) lives there (--ipc host) with ~1 GiB spare" ;; esac
  fi
  pc_shm_suffix="glm53-r${NODE_RANK}-${pc_port}"
  pc_l2=$(printf '{"type":"fs_native","base_path":"%s","num_workers":%s,"use_odirect":false,"max_capacity_gb":%s,"eviction":{"eviction_policy":"LRU","trigger_watermark":0.8,"eviction_ratio":0.2}}' \
    "$pc_namespace" "$PREFIX_CACHE_LMC_L2_WORKERS" "$PREFIX_CACHE_DISK_GB")
  lmc_health_url="http://127.0.0.1:$(( pc_port + 1 ))/healthcheck"
  # CPU-only service: no GPUs, CUDA hidden. Same image (the lmcache CLI and
  # its torch live there), host network/IPC so vLLM maps the L1 SHM pool.
  lmc_server_cmd=(
    "$CONTAINER_RUNTIME" run -d
    --name "$lmc_container"
    --pull never
    --network host
    --ipc host
    --ulimit memlock=-1:-1
    --stop-timeout 10
    -v "$PREFIX_CACHE_HOST_PATH:/prefix-cache"
    -e CUDA_VISIBLE_DEVICES=
    -e CUDA_MODULE_LOADING=LAZY
    -e LD_PRELOAD="$LD_PRELOAD"
    --entrypoint /opt/venv/bin/lmcache
    "$SERVING_IMAGE"
    server
    --instance-id "glm53-r${NODE_RANK}"
    --host "$VLLM_HOST_IP" --port "$pc_port"
    --http-host 127.0.0.1 --http-port $(( pc_port + 1 ))
    --prometheus-port $(( pc_port + 2 ))
    --chunk-size "$pc_chunk"
    --supported-transfer-mode engine_driven
    --separate-object-groups
    --l1-size-gb "$PREFIX_CACHE_LMC_L1_GB" --l1-init-size-gb "$PREFIX_CACHE_LMC_L1_GB"
    --no-l1-use-lazy --shm-name "$pc_shm_suffix"
    --max-gpu-workers 1 --max-cpu-workers "$PREFIX_CACHE_LMC_CPU_WORKERS"
    --eviction-policy LRU --l2-prefetch-policy retain --emergency-evict-for-prefetch
    --hash-algorithm blake3 --max-workers 8
    --l2-adapter "$pc_l2"
  )
  pc_kv_transfer=$(printf '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.server_urls":["tcp://%s:%s","tcp://%s:%s"],"lmcache.mp.mp_transfer_mode":"engine_driven"}}' \
    "$MASTER_ADDR" "$pc_port" "$PREFIX_CACHE_LMC_RANK1_ADDR" "$pc_port")
  extra_args+=(--kv-transfer-config "$pc_kv_transfer" --prefix-cache-retention-interval "$pc_chunk")
  prefix_cache_desc="lmcache -> $PREFIX_CACHE_HOST_PATH (L1 ${PREFIX_CACHE_LMC_L1_GB} GiB shm pinned/node -- take it off the KV pin; L2 ${PREFIX_CACHE_DISK_GB} GiB, chunk ${pc_chunk}, servers ${MASTER_ADDR}+${PREFIX_CACHE_LMC_RANK1_ADDR}:${pc_port})"
  pc_ram_gib=$PREFIX_CACHE_LMC_L1_GB
fi

# KV-pin budget. The profile pin (19377663936) leaves rank 0 ~2 GiB
# MemAvailable under the full bench (2026-10-09). A connector adds its host
# RAM (lmcache L1, sparkcache staging) plus the REPLAYSSM=0 reservation
# (~188 MiB/rank) to the same unified pool, so the pin must give that back.
if [ "$PREFIX_CACHE" != none ] && [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
  pc_pin_max=$(awk -v ref=19377663936 -v g="$pc_ram_gib" 'BEGIN{printf "%d", ref - (g + 0.2) * 1073741824}')
  [ "$KV_CACHE_MEMORY_BYTES" -le "$pc_pin_max" ] \
    || warn "PREFIX_CACHE=$PREFIX_CACHE adds ~${pc_ram_gib} GiB host RAM + ~0.2 GiB (REPLAYSSM=0) to the unified pool; KV_CACHE_MEMORY_BYTES=$KV_CACHE_MEMORY_BYTES would eat into the ~2 GiB rank 0 keeps at the profile pin. Use <= $pc_pin_max and re-check MemAvailable under load"
fi
# ===== END prefix cache =====

# ===== BEGIN shared block: b12x loader io_uring (identical in every *_pair-serve.sh) =====
# LOAD_FORMAT=b12x reads weights through an O_DIRECT io_uring ring (b12x
# 1ec67ef6). Three things must hold, and each fails differently:
#   1. liburing in the IMAGE -- b12x builds the reader at first use and treats
#      liburing as optional: "io_uring bounce support is unavailable".
#      (build-spark-cu132.sh PATCH_IO_URING bakes it.)
#   2. io_uring_setup/enter/register allowed by the CONTAINER seccomp. Docker
#      >= 25's default profile blocks them, and an image cannot relax its own
#      seccomp (the runtime installs the filter before the entrypoint), so the
#      profile is passed from here: one shared copy at the deployments root.
#   3. HOST sysctl kernel.io_uring_disabled=0.
# SECCOMP_PROFILE (env file; usually left out):
#   unset/empty = auto: with LOAD_FORMAT=b12x use <deployments>/seccomp-io-uring.json
#                 (the parent of this script's folder); other loaders pass nothing
#   <path>      = that profile (absolute, or relative to this script's folder)
#   none        = pass nothing (the docker daemon default already allows io_uring)
#   unconfined  = no syscall filtering at all (last resort)
: "${SECCOMP_PROFILE:=}"
: "${SCRIPT_DIR:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
seccomp_args=()
seccomp_desc="runtime default"
seccomp_path=""
case "$SECCOMP_PROFILE" in
  none) seccomp_desc="runtime default (SECCOMP_PROFILE=none)" ;;
  unconfined)
    seccomp_args=(--security-opt seccomp=unconfined); seccomp_desc=unconfined
    warn "SECCOMP_PROFILE=unconfined disables syscall filtering for the whole container; prefer the shared seccomp-io-uring.json" ;;
  "") [ "$LOAD_FORMAT" != b12x ] || seccomp_path="$SCRIPT_DIR/../seccomp-io-uring.json" ;;
  /*) seccomp_path=$SECCOMP_PROFILE ;;
  *)  seccomp_path="$SCRIPT_DIR/$SECCOMP_PROFILE" ;;
esac
if [ -n "$seccomp_path" ]; then
  [ -f "$seccomp_path" ] || die "seccomp profile not found: $seccomp_path -- LOAD_FORMAT=b12x needs io_uring allowed in the container. Put the shared seccomp-io-uring.json in the deployments root ($(cd "$SCRIPT_DIR/.." && pwd)/), point SECCOMP_PROFILE at a copy, or set SECCOMP_PROFILE=none if the docker daemon default already allows io_uring"
  seccomp_path="$(cd "$(dirname "$seccomp_path")" && pwd)/$(basename "$seccomp_path")"
  for sc in io_uring_setup io_uring_enter io_uring_register; do
    grep -q "\"$sc\"" "$seccomp_path" \
      || die "seccomp profile $seccomp_path does not list $sc -- wrong file? (expected vllm scripts/seccomp/spark-io-uring.json)"
  done
  seccomp_args=(--security-opt "seccomp=$seccomp_path"); seccomp_desc=$seccomp_path
fi
# Probe the image with exactly the seccomp the serve container will get.
# io_uring_setup is syscall 425 on aarch64 and x86_64; with NULL params an
# allowed kernel answers EFAULT, a blocking seccomp (or sysctl) answers EPERM.
if [ "$LOAD_FORMAT" = b12x ] \
   && command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1 \
   && "$CONTAINER_RUNTIME" image inspect "$SERVING_IMAGE" >/dev/null 2>&1; then
  "$CONTAINER_RUNTIME" run --rm --entrypoint sh "$SERVING_IMAGE" -c 'pkg-config --exists liburing' >/dev/null 2>&1 \
    || die "LOAD_FORMAT=b12x: $SERVING_IMAGE has no liburing development files, so the b12x loader dies with \"io_uring bounce support is unavailable\". Rebuild with PATCH_IO_URING=on (build-spark-cu132.sh), or use LOAD_FORMAT=instanttensor"
  uring_probe="$("$CONTAINER_RUNTIME" run --rm "${seccomp_args[@]}" --entrypoint /opt/venv/bin/python "$SERVING_IMAGE" -c '
import ctypes, errno
libc = ctypes.CDLL(None, use_errno=True); libc.syscall(425, 0, 0)
print("BLOCKED" if ctypes.get_errno() == errno.EPERM else "OK")' 2>&1 || true)"
  case "$uring_probe" in
    *BLOCKED*) die "io_uring is BLOCKED in the container under seccomp '$seccomp_desc' (EPERM). Check the profile, and the host: sysctl kernel.io_uring_disabled must be 0" ;;
    *OK*) ;;
    *) warn "io_uring probe inconclusive (${uring_probe:-no output}); the load will tell" ;;
  esac
fi
# ===== END shared block: b12x loader io_uring =====

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
  "${seccomp_args[@]}"   # b12x loader io_uring (shared block above)
  "${mount_args[@]}"
  --env-file "$env_file"
  # The image bakes VLLM_PCIE_ALLREDUCE_BACKEND=cpp (pre-rename); vLLM
  # validates it with choices ['b12x'] at worker start even when the PCIe
  # path is off. -e beats both --env-file and the baked ENV.
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
  "${precision_env_args[@]}"
  "${dcp_env_args[@]}"
  "${profile_env_args[@]}"
  "${instanttensor_env_args[@]}"
  --entrypoint /opt/venv/bin/vllm
  "$SERVING_IMAGE"
  serve "$model_container_path"

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

  # --- model / kernels (FP4-CSF: ModelOpt recipes own the routed experts) --
  --dtype bfloat16
  --quantization modelopt_mixed
  --load-format "$LOAD_FORMAT"
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
printf '  checkpoint:              %s (hf layout%s, %s shards, %s GB)\n' "$MODEL_HOST_PATH" \
  "$(case "$csf_revision" in -) ;; mixed) printf ', MIXED revisions' ;; *) printf ', revision %s' "${csf_revision:0:12}" ;; esac)" \
  "$csf_shards" "$(awk -v b="$csf_bytes" 'BEGIN{printf "%.1f", b/1e9}')"
[ -z "$CHECKPOINT_REVISION" ] || printf '  checkpoint revision:     %s (CHECKPOINT_REVISION %s)\n' "${csf_revision:0:12}" "$CHECKPOINT_REVISION"
printf '  checkpoint identity:     %s (must match on both ranks%s)\n' "$csf_identity" \
  "$(case "$csf_lil" in match) printf '; = blackwell-llm-docker pinned content' ;; mismatch) printf '; DIFFERS from the pinned content' ;; esac)"
printf '  expert input scales:     %s (A4 prefill %s)\n' "$csf_input_scales" \
  "$(if [ "$PREFILL_ACTIVATIONS" = a16 ]; then echo off; elif [ "$csf_input_scales" -gt 0 ]; then echo possible; else echo 'NOT possible: B12X runs A16'; fi)"
printf '  vision tower:            %s\n' "$(if [ "$LANGUAGE_MODEL_ONLY" = 1 ]; then echo 'off (LANGUAGE_MODEL_ONLY=1)'; elif [ "$csf_vision_stored" = 1 ]; then echo 'stored recipes (checkpoint)'; else echo 'BF16 (no vision recipes in the checkpoint)'; fi)"
printf '  vocabulary heads:        verifier %s; draft %s\n' \
  "$([ "${VLLM_MXFP8_LM_HEAD:-0}" = 1 ] && echo MXFP8 || echo BF16)" \
  "$(if [ "${VLLM_MTP_NVFP4_LM_HEAD:-1}" = 1 ]; then echo 'own NVFP4 head (path A)'; elif [ "$draft_head" = nvfp4 ]; then echo 'NVFP4 copy of the BF16 target head (path B)'; else echo 'shares the verifier head'; fi)"
[ "$SPECULATOR" != dflash2 ] || printf '  draft model:             %s\n' "$DFLASH_MODEL_HOST_PATH"
printf '  image:                   %s (%s)\n' "$SERVING_IMAGE" "$image_desc"
printf '  CSF load:                --quantization modelopt_mixed --load-format %s%s\n' "$LOAD_FORMAT" \
  "$(case "$LOAD_FORMAT" in instanttensor) printf ' (buffer %s, depth %s, concurrency %s, chunk %s)' "$INSTANTTENSOR_BUFFER_SIZE" "$INSTANTTENSOR_IO_DEPTH" "$INSTANTTENSOR_CONCURRENCY" "$INSTANTTENSOR_CHUNK_SIZE" ;; esac)"
printf '  seccomp:                 %s\n' "$seccomp_desc"
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
printf '  prefix cache:            %s\n' "$prefix_cache_desc"
printf '  fabric profile:          %s (NCCL %s; RoCEnante %s)\n' "$FABRIC_PROFILE" "$NCCL_IB_HCA" "${B12X_ROCE_HCA:-follows NCCL_IB_HCA}"
printf '  collectives:             %s\n' "$roce_desc"
printf '  tokenizer backend:       %s\n' "$([ "$VLLM_USE_FASTOKENS" = 1 ] && echo fastokens || echo 'HF tokenizers (standard)')"
printf '  chat template:           %s\n' "${CHAT_TEMPLATE_HOST_PATH:-checkpoint default}"
printf '  API_KEY:                 %s\n' "$([ -n "$API_KEY" ] && echo 'set (Bearer required)' || echo 'none (open port)')"
printf '  command:'
printf ' %q' "${command[@]}"
printf '\n'
if [ "${#lmc_server_cmd[@]}" -gt 0 ]; then
  printf '  lmcache server:'
  printf ' %q' "${lmc_server_cmd[@]}"
  printf '\n'
fi

[ "$mode" = --run ] || exit 0

# --------------------------------------------------------------------- run

require_runtime
"$CONTAINER_RUNTIME" image inspect "$SERVING_IMAGE" >/dev/null 2>&1 \
  || die "pinned image is not present; build or load it before launching: $SERVING_IMAGE"
if "$CONTAINER_RUNTIME" container inspect "$container_name" >/dev/null 2>&1; then
  die "container already exists; remove it intentionally before relaunch: $container_name"
fi

# PREFIX_CACHE=lmcache: the node-local server must answer before vLLM's
# connector connects (and, on rank 1, before rank 0 starts: rank 0's
# scheduler queries both servers). Start rank 1 first, as always.
if [ "${#lmc_server_cmd[@]}" -gt 0 ]; then
  command -v curl >/dev/null 2>&1 || die "PREFIX_CACHE=lmcache needs curl on the host for the lmcache server health check"
  if "$CONTAINER_RUNTIME" container inspect "$lmc_container" >/dev/null 2>&1; then
    die "container already exists; remove it intentionally before relaunch (--down removes both): $lmc_container"
  fi
  "${lmc_server_cmd[@]}" >/dev/null
  printf 'waiting for %s (%s) ' "$lmc_container" "$lmc_health_url"
  pc_deadline=$(( $(date +%s) + ${PREFIX_CACHE_LMC_START_TIMEOUT:-120} ))
  until curl -fsS "$lmc_health_url" >/dev/null 2>&1; do
    if ! "$CONTAINER_RUNTIME" container inspect -f '{{.State.Running}}' "$lmc_container" 2>/dev/null | grep -q true \
       || [ "$(date +%s)" -ge "$pc_deadline" ]; then
      printf 'FAILED\n'
      "$CONTAINER_RUNTIME" logs --tail 40 "$lmc_container" >&2 || true
      die "lmcache server did not become healthy; container left for inspection: $lmc_container (--down removes it)"
    fi
    printf '.'; sleep 2
  done
  printf ' ready\n'
fi

if [ "$fresh_follow" = 1 ]; then
  "${command[@]}"
  printf '%s started; following logs (Ctrl-C detaches, the container keeps running)\n' "$container_name"
  exec "$CONTAINER_RUNTIME" logs -f "$container_name"
else
  exec "${command[@]}"
fi
