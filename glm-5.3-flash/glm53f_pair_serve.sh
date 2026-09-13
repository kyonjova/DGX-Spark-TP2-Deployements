#!/usr/bin/env bash
# glm53_pair_serve.sh
#
# Validate or start one rank of a two-node DGX Spark pair serving
# GLM-5.3-Flash-NVFP4 under vLLM (dev/jovian-judgement + B12X/SparkInfer),
# with either the built-in MTP head or the DFlash2 draft model as speculator.
#
# The per-rank env file is both the host launch contract and the container
# environment, so operator-facing paths and serving values have one source of
# truth. Every value is validated before a container is started: unresolved
# placeholders, CRLF line endings, missing model bytes, occupied ports, a
# VLLM_HOST_IP that is not local, LD_PRELOAD paths absent from the image, and
# known-bad combinations all fail at --check time rather than minutes into a
# boot.
#
# Usage:
#     ./glm53_pair_serve.sh --check   rank-0.env
#     ./glm53_pair_serve.sh --run     rank-0.env
#     ./glm53_pair_serve.sh --restart rank-0.env
#     ./glm53_pair_serve.sh --logs    rank-0.env
#     ./glm53_pair_serve.sh --status  rank-0.env
#     ./glm53_pair_serve.sh --down    rank-0.env
#     ./glm53_pair_serve.sh --clear   rank-0.env
#     ./glm53_pair_serve.sh --fresh   rank-0.env # drop page cache, then --run, then follow logs
#
# Anything after ENV_FILE is appended verbatim to the vllm serve argv:
#     ./glm53_pair_serve.sh --run rank-0.env --logprobs-mode processed_logprobs
#
# Start rank 1 (headless, waits for rank 0) before rank 0.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: glm53_pair_serve.sh [MODE] ENV_FILE [extra vllm args...]

  --check     validate the env file and print the launch command (default)
  --run       validate, then start the container detached
  --restart   stop and remove this rank's container, then run
  --down      stop and remove this rank's container
  --logs      follow this rank's container logs
  --verify    grep this rank's log for the qualified startup markers (+ /health on rank 0)
  --status    show this rank's container state
  --clear     wipe CACHE_HOST_PATH contents (container must not exist)
  --fresh     drop page cache (sync + sudo), then --run, then follow logs
EOF
}

die() {
  echo "glm53 pair launcher: $*" >&2
  exit 20
}

warn() {
  echo "glm53 pair launcher: warning: $*" >&2
}

# Reclaim page cache: on GB10 CUDA free memory tracks MemFree, so page cache
# left by a previous model load counts AGAINST vLLM's own startup gate (the
# mem preflight below names the culprit). Used by the --fresh mode.
drop_page_cache() {
  if [ ! -e /proc/sys/vm/drop_caches ]; then
    die "cannot drop page cache: /proc/sys/vm/drop_caches does not exist (not Linux?)"
  fi
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

# --fresh reclaims page cache BEFORE anything else: the mem preflight below
# reads MemFree, and on GB10 CUDA free memory tracks it. Then it behaves
# exactly like --run, except the container is started without exec so the
# launcher can follow the logs afterwards.
if [ "$mode" = --fresh ]; then
  drop_page_cache
  mode=--run
  fresh_follow=1
fi

# CRLF breaks both the shell source below and docker --env-file, silently and
# in different ways. Catch it before either consumer sees it.
if grep -qU $'\r' "$env_file" 2>/dev/null; then
  die "environment file has CRLF line endings; convert with: sed -i 's/\r$//' $env_file"
fi

if grep -Ev '^[[:space:]]*(#|$)' "$env_file" \
   | grep -Eq '<[A-Za-z0-9_]+>|REPLACE_WITH_'; then
  die "environment file contains unresolved placeholders: $env_file"
fi

# Values may not contain whitespace: the file is both sourced by this shell
# (a space ends the assignment and runs the remainder as a command) and passed
# to docker --env-file (which keeps trailing comments as part of the value).
if grep -nE '^[A-Z][A-Z0-9_]*=[^#]*[[:space:]]' "$env_file" | grep -vE '^[0-9]+:[A-Z][A-Z0-9_]*=[[:space:]]*$' | head -3 | grep -q .; then
  grep -nE '^[A-Z][A-Z0-9_]*=[^#]*[[:space:]]' "$env_file" | grep -vE '^[0-9]+:[A-Z][A-Z0-9_]*=[[:space:]]*$' | head -3 >&2
  die "environment file has values containing whitespace or trailing comments (lines above); quote nothing, put comments on their own line"
fi

# shellcheck disable=SC1090
. "$env_file"

# ------------------------------------------------- container management modes
# These need only the rank and runtime, so they run before full validation.

: "${CONTAINER_RUNTIME:=docker}"
: "${CONTAINER_NAME_SUFFIX:=}"
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
}

container_clear() {
  local target=${CACHE_HOST_PATH-} count

  # Refuse anything that is not plainly a cache directory under a home or data
  # path. rm -rf against a bad value here would be unrecoverable.
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
  { find "$target" -mindepth 1 -maxdepth 1 -printf '  %f\n' 2>/dev/null || true; } \
    | head -20 || true

  if [ "${CLEAR_ASSUME_YES:-0}" != 1 ]; then
    printf 'type "clear" to confirm: '
    read -r reply
    [ "$reply" = clear ] || die "aborted"
  fi

  find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  printf 'cleared %s\n' "$target"
  printf 'note: next boot recompiles torch.compile, CuTeDSL, Triton and FlashInfer artifacts\n'
}

case "$mode" in
  --clear)
    container_clear
    exit 0
    ;;
  --down)
    container_down
    exit 0
    ;;
  --logs)
    require_runtime
    exec "$CONTAINER_RUNTIME" logs -f "$mgmt_container"
    ;;
  --verify)
    require_runtime
    # Marker set from the rtx6kpro r7 runbook's "Verify startup" section, plus
    # the page-split and KV lines that matter on the pair. Absence of a marker
    # you expect (e.g. B12xMxfp8 under dflash2) means the wrong path was taken.
    "$CONTAINER_RUNTIME" logs "$mgmt_container" 2>&1 | grep -E \
      'speculative_config|Using .* all-reduce backends|RoCEnante|B12X_ROCENANTE|kda_prefill|KDA prefill|B12xMxfp8|HUMMING|Humming|FlashAttention version 2|FlashKDA|split GLM-5.3 cache pages|DFlash draft KV layers|physical page sizes|Add [0-9]+ padding layers|attention block size|Available KV cache memory|GPU KV cache size|Graph capturing finished|cudagraph_mode=|Application startup complete' \
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
  local name=$1 value
  value=${!name-}
  [ -n "$value" ] || die "required value is empty: $name"
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
  local name=$1 value
  require_positive_integer "$name"
  value=${!name-}
  [ "$((10#$value))" -le 65535 ] || die "$name must be in 1..65535: $value"
}

require_bool() {
  local name=$1 value
  value=${!name-}
  case "$value" in
    0|1) ;;
    *) die "$name must be 0 or 1: $value" ;;
  esac
}

require_unit_fraction() {
  local name=$1 value
  value=${!name-}
  awk -v v="$value" 'BEGIN{ exit !(v+0 > 0 && v+0 <= 1 && v ~ /^[0-9]*\.?[0-9]+$/) }' \
    || die "$name must be a fraction in (0,1]: $value"
}

# ------------------------------------------------------- serving defaults
# Every value below may be set in the env file; these are the fallbacks.
# Serving flags mirror serve-glm53-flash-nvfp4.sh from dev/jovian-judgement --
# the canonical single-node recipe for this checkpoint -- with the two-node
# topology and DGX Spark fabric handling carried over from the validated
# DeepSeek pair launcher.

: "${SERVED_MODEL_NAME:=zai-org/GLM-5.3-Flash}"
: "${SPECULATOR:=mtp}"                 # mtp | dflash2 | none
: "${GPU_MEMORY_UTILIZATION:=0.90}"   # measured stable through profiling on the pair; GB10 ceiling ~0.913
: "${KV_CACHE_MEMORY_BYTES:=}"         # empty = let vLLM profile and choose
: "${KV_CACHE_DTYPE:=fp8}"
: "${DFLASH_KV_CACHE_DTYPE:=auto}"     # draft KV; auto (BF16) is the qualified value
: "${QUANTIZATION:=modelopt_mixed}"    # `auto` omits the flag (checkpoint self-describes)
: "${PREFILL_SCHEDULE_INTERVAL:=8}"    # admit new prefill every N engine steps; inert before #546 (r8), then stops decode starvation
: "${COMPILATION_LEVEL:=}"             # empty = no -O flag; 0-3 passes -O<N> (torch.compile level) -- A/B only
: "${GENERATION_CONFIG:=auto}"         # auto = model's generation_config.json; vllm = engine defaults
: "${SAMPLING_TEMPERATURE:=}"          # empty = checkpoint/engine default; r30 GLM launcher supplies 1.0 (publisher generation_config)
: "${SAMPLING_TOP_P:=}"                # empty = default; r30 supplies 0.95; fixed top-k stays off (GLM does not inherit Qwen's 20)
: "${SAMPLING_TOP_K:=}"                # empty = default (off); integer
: "${SAMPLING_MIN_P:=}"                # empty = default (0)
: "${SAMPLING_REPETITION_PENALTY:=}"   # empty = default (1.0)
: "${REASONING_EFFORT:=}"              # empty = template default; low|high|max via --default-chat-template-kwargs (r30 default: high)
: "${CLEAR_THINKING:=}"                # empty = template default; 0 keeps client-supplied historical reasoning in the prompt (r30 agent profile); 1 strips it
: "${KDA_PREFILL_BACKEND:=}"           # empty = engine default; b12x|flashkda|triton|auto (needs vllm pin >= 54371894e, 2026-09-03)
: "${GDN_DECODE_KERNEL:=}"             # empty = engine default; b12x|cuda|triton (explicit KDA decode kernel; Luke's TP2 launcher pins b12x)
: "${FAIRNESS_ENGINE:=}"               # empty = none; compute_share|micro_slicing (r26 scheduler; pairs with PREFILL_COMPUTE_SHARE)
: "${RECURRENT_CHECKPOINT_POLICY:=}"   # empty = engine default (auto); request_boundaries|aligned (vllm pin >= 52d9977b, 2026-09-05)
: "${PREFILL_COMPUTE_SHARE:=}"         # fraction of contended execution for prefill; r26 qualifies 0.40 with compute_share
: "${BLOCK_SIZE:=256}"                 # paged-KV block; Luke's TP2 launcher uses 16 on the packed-cache (r22) tree
: "${LIMIT_MM:=1}"                     # 1 = pass --limit-mm-per-prompt from MM_IMAGES/MM_VIDEOS; 0 = omit it (engine default: 999 per modality, bounded only by MAX_MODEL_LEN)
: "${MM_PROCESSOR_CACHE_GB:=}"         # empty = engine default (4 GiB of host RAM = unified memory on Spark); 0 disables
: "${MM_ENCODER_TP_MODE:=}"            # empty = engine default; data = each rank encodes its own images (no encoder TP collectives)
: "${CHAT_TEMPLATE_HOST_PATH:=}"       # optional custom .jinja (e.g. reasoning default high)
: "${INSTANTTENSOR_COPY:=auto}"        # 0 = --model-loader-extra-config instanttensor_copy:false (bounded loading, needs 2026-08-29+ pin)
: "${TORCH_PROFILE_HOST_DIR:=}"        # set to a host dir to enable /start_profile + /stop_profile
: "${CONTAINER_MEMORY_GB:=}"           # cgroup cap so a runaway load cannot OOM the host (Luke uses 108)
: "${ADAPTIVE_SPECULATIVE_TOKENS:=0}"  # mtp only; needs vllm pin >= e10536aa (2026-08-29)
: "${ADAPTIVE_SPECULATIVE_TOKENS_WINDOW:=32}"
# INITIAL defaults to min(3, depth) below once depth is known.
# TP=2 InstantTensor staging bounds from the canonical launcher (bounds the
# checkpoint-loading memory peak; harmless on older images).
: "${INSTANTTENSOR_BUFFER_SIZE:=1342177280}"  # 1.25 GiB: the largest tensor is 1,268,776,960 B; 64 MiB just got auto-enlarged every boot
: "${INSTANTTENSOR_IO_DEPTH:=3}"
: "${INSTANTTENSOR_CONCURRENCY:=1}"
: "${INSTANTTENSOR_CHUNK_SIZE:=8388608}"
: "${API_KEY:=}"                       # empty = open port; set to require Authorization: Bearer <key> on rank 0 (from the template)
: "${ENABLE_FLASHINFER_AUTOTUNE:=0}"   # 0 = --no-enable-flashinfer-autotune (canonical GLM recipe; B12X owns attention/MoE/linear); 1 = tune FlashInfer kernels at startup
: "${MOE_BACKEND:=b12x}"               # b12x (qualified; the engine default since r34/#728) | humming | auto
: "${ATTENTION_BACKEND:=B12X}"          # B12X (qualified GLM sparse-MLA + GDN path) | auto
: "${LINEAR_BACKEND:=b12x}"            # restrict linear kernels to the b12x set
: "${LANGUAGE_MODEL_ONLY:=1}"          # skip vision encoder cache + max-video profile buffers
: "${MM_IMAGES:=4}"                    # per-prompt image cap when LANGUAGE_MODEL_ONLY=0
: "${MM_VIDEOS:=1}"                    # per-prompt video cap when LANGUAGE_MODEL_ONLY=0
: "${MAX_MODEL_LEN:=229376}"          # measured pair ceiling at util 0.90 is 235,008 tokens; this leaves ~11% margin
: "${MAX_NUM_SEQS:=8}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"   # qualified value; 8192 doubles the profiling activation peak
: "${MAX_CUDAGRAPH_CAPTURE_SIZE:=96}"
: "${CUDAGRAPH_MODE:=FULL_AND_PIECEWISE}" # FULL auto-downgrades to FULL_DECODE_ONLY on the GDN backend, leaving prefill EAGER; this adds piecewise prefill graphs
: "${LOAD_FORMAT:=instanttensor}"
: "${ENABLE_PREFIX_CACHING:=1}"
: "${ENABLE_CHUNKED_PREFILL:=1}"
: "${TRUST_REMOTE_CODE:=0}"
: "${CONTAINER_RUNTIME:=docker}"
: "${CONTAINER_NAME_SUFFIX:=}"
: "${SERVING_IMAGE:=local/vllm:glm53-jovian-judgement-b12x-cu132-sm121}"
: "${SHM_SIZE:=16g}"

# Accept MTP=<n> as an alias for NUM_SPECULATIVE_TOKENS under SPECULATOR=mtp,
# matching the single-node run commands this recipe was derived from.
if [ -n "${MTP-}" ] && [ -z "${NUM_SPECULATIVE_TOKENS-}" ]; then
  NUM_SPECULATIVE_TOKENS=$MTP
fi

case "$SPECULATOR" in
  mtp)     : "${NUM_SPECULATIVE_TOKENS:=5}" ;;   # branch default
  dflash2) : "${NUM_SPECULATIVE_TOKENS:=7}" ;;
  none)    : "${NUM_SPECULATIVE_TOKENS:=0}" ;;
  *) die "SPECULATOR must be mtp, dflash2, or none: $SPECULATOR" ;;
esac

case "$ADAPTIVE_SPECULATIVE_TOKENS" in 0|1) : ;; *) die "ADAPTIVE_SPECULATIVE_TOKENS must be 0 or 1" ;; esac
if [ "$ADAPTIVE_SPECULATIVE_TOKENS" = 1 ]; then
  [ "$SPECULATOR" = mtp ] && [ "$NUM_SPECULATIVE_TOKENS" -gt 0 ] \
    || die "ADAPTIVE_SPECULATIVE_TOKENS requires SPECULATOR=mtp with positive depth"
  if [ -z "${ADAPTIVE_SPECULATIVE_TOKENS_INITIAL:-}" ]; then
    if [ "$NUM_SPECULATIVE_TOKENS" -lt 3 ]; then ADAPTIVE_SPECULATIVE_TOKENS_INITIAL=$NUM_SPECULATIVE_TOKENS
    else ADAPTIVE_SPECULATIVE_TOKENS_INITIAL=3; fi
  fi
  require_positive_integer ADAPTIVE_SPECULATIVE_TOKENS_INITIAL
  require_positive_integer ADAPTIVE_SPECULATIVE_TOKENS_WINDOW
  [ "$ADAPTIVE_SPECULATIVE_TOKENS_INITIAL" -le "$NUM_SPECULATIVE_TOKENS" ] \
    || die "ADAPTIVE_SPECULATIVE_TOKENS_INITIAL must not exceed NUM_SPECULATIVE_TOKENS"
fi

require_positive_integer PREFILL_SCHEDULE_INTERVAL
case "$COMPILATION_LEVEL" in ""|0|1|2|3) : ;; *) die "COMPILATION_LEVEL must be empty or 0-3: $COMPILATION_LEVEL" ;; esac
case "$GENERATION_CONFIG" in auto|vllm) : ;; *) die "GENERATION_CONFIG must be auto or vllm: $GENERATION_CONFIG" ;; esac
case "$KDA_PREFILL_BACKEND" in ""|auto|triton|flashkda|b12x) : ;; *) die "KDA_PREFILL_BACKEND must be empty, auto, triton, flashkda, or b12x: $KDA_PREFILL_BACKEND" ;; esac
case "$GDN_DECODE_KERNEL" in ""|b12x|cuda|triton) : ;; *) die "GDN_DECODE_KERNEL must be empty, b12x, cuda, or triton: $GDN_DECODE_KERNEL" ;; esac
case "$FAIRNESS_ENGINE" in ""|compute_share|micro_slicing) : ;; *) die "FAIRNESS_ENGINE must be empty, compute_share, or micro_slicing: $FAIRNESS_ENGINE" ;; esac
case "$LOAD_FORMAT" in
  instanttensor|fastsafetensors|auto|safetensors) : ;;
  b12x)
    case ",${VLLM_PLUGINS-}," in
      *,b12x_loader,*) : ;;
      *) die "LOAD_FORMAT=b12x needs the loader plugin: set VLLM_PLUGINS=b12x_loader (vllm pin >= 9e500700, 2026-09-06)" ;;
    esac ;;
  *) die "LOAD_FORMAT must be instanttensor, fastsafetensors, b12x, auto, or safetensors: $LOAD_FORMAT" ;;
esac
for name in SAMPLING_TEMPERATURE SAMPLING_TOP_P SAMPLING_TOP_K SAMPLING_MIN_P SAMPLING_REPETITION_PENALTY; do
  v=${!name-}
  [ -z "$v" ] || awk -v v="$v" 'BEGIN{ exit !(v ~ /^[0-9]*\.?[0-9]+$/) }' || die "$name must be a number: $v"
done
case "$REASONING_EFFORT" in ""|low|high|max) : ;; *) die "REASONING_EFFORT must be empty, low, high, or max: $REASONING_EFFORT" ;; esac
case "$CLEAR_THINKING" in ""|0|1) : ;; *) die "CLEAR_THINKING must be empty, 0, or 1: $CLEAR_THINKING" ;; esac
case "$RECURRENT_CHECKPOINT_POLICY" in ""|auto|request_boundaries|aligned) : ;; *) die "RECURRENT_CHECKPOINT_POLICY must be empty, auto, request_boundaries, or aligned: $RECURRENT_CHECKPOINT_POLICY" ;; esac
if [ -n "$PREFILL_COMPUTE_SHARE" ]; then
  [ "$FAIRNESS_ENGINE" = compute_share ] || die "PREFILL_COMPUTE_SHARE requires FAIRNESS_ENGINE=compute_share"
  awk -v v="$PREFILL_COMPUTE_SHARE" 'BEGIN{ exit !(v+0 > 0 && v+0 < 1 && v ~ /^[0-9]*\.?[0-9]+$/) }' \
    || die "PREFILL_COMPUTE_SHARE must be a fraction in (0,1): $PREFILL_COMPUTE_SHARE"
fi
require_positive_integer BLOCK_SIZE
case "$MM_PROCESSOR_CACHE_GB" in ""|[0-9]*) : ;; *) die "MM_PROCESSOR_CACHE_GB must be a number: $MM_PROCESSOR_CACHE_GB" ;; esac
case "$MM_ENCODER_TP_MODE" in ""|data|weights) : ;; *) die "MM_ENCODER_TP_MODE must be empty, data, or weights: $MM_ENCODER_TP_MODE" ;; esac
if [ -n "$CHAT_TEMPLATE_HOST_PATH" ]; then
  [ -f "$CHAT_TEMPLATE_HOST_PATH" ] || die "CHAT_TEMPLATE_HOST_PATH does not exist: $CHAT_TEMPLATE_HOST_PATH"
fi

# ------------------------------------------------------------ launch checks

for name in \
  NODE_RANK MASTER_ADDR MODEL_HOST_PATH CACHE_HOST_PATH API_PORT MASTER_PORT \
  MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
  LD_PRELOAD VLLM_NCCL_SO_PATH NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME \
  VLLM_HOST_IP NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_IB_HCA \
  NCCL_IB_GID_INDEX NCCL_IB_SUBNET_AWARE_ROUTING NCCL_IB_MERGE_NICS \
  NCCL_PROTO NCCL_P2P_LEVEL NCCL_CROSS_NIC NCCL_CUMEM_ENABLE \
  NCCL_IGNORE_CPU_AFFINITY CUTE_DSL_ARCH \
  VLLM_ENABLE_PCIE_ALLREDUCE VLLM_B12X_MOE_FP4_FORCE_A16; do
  require_value "$name"
done

case "$NODE_RANK" in
  0|1) ;;
  *) die "NODE_RANK must be 0 or 1: $NODE_RANK" ;;
esac

require_directory MODEL_HOST_PATH
require_directory CACHE_HOST_PATH
[ -r "$MODEL_HOST_PATH" ] || die "MODEL_HOST_PATH is not readable: $MODEL_HOST_PATH"
# An existing-but-wrong mount fails obscurely inside the container, and
# HF_HUB_OFFLINE=1 removes any chance of recovery. Check for actual model bytes.
[ -f "$MODEL_HOST_PATH/config.json" ] \
  || die "MODEL_HOST_PATH has no config.json; wrong directory or a broken mount: $MODEL_HOST_PATH"
ls "$MODEL_HOST_PATH"/*.safetensors >/dev/null 2>&1 \
  || die "MODEL_HOST_PATH contains no weight files (*.safetensors): $MODEL_HOST_PATH"
[ -w "$CACHE_HOST_PATH" ] || die "CACHE_HOST_PATH is not writable: $CACHE_HOST_PATH"

if [ "$SPECULATOR" = dflash2 ]; then
  require_value DFLASH_MODEL_HOST_PATH
  require_directory DFLASH_MODEL_HOST_PATH
  [ -f "$DFLASH_MODEL_HOST_PATH/config.json" ] \
    || die "DFLASH_MODEL_HOST_PATH has no config.json: $DFLASH_MODEL_HOST_PATH"
  ls "$DFLASH_MODEL_HOST_PATH"/*.safetensors >/dev/null 2>&1 \
    || die "DFLASH_MODEL_HOST_PATH contains no weight files: $DFLASH_MODEL_HOST_PATH"

  # Optional but recommended: pin the draft by content. The r7 qualification
  # publishes sha256 c033e03d... for the MXFP8 draft's model.safetensors; a
  # mismatch means you are not serving the qualified weights.
  if [ -n "${DFLASH_WEIGHTS_SHA256-}" ]; then
    printf '%s' "$DFLASH_WEIGHTS_SHA256" | grep -Eq '^[0-9a-f]{64}$' \
      || die "DFLASH_WEIGHTS_SHA256 must be 64 lowercase hex chars"
    [ -f "$DFLASH_MODEL_HOST_PATH/model.safetensors" ] \
      || die "DFLASH_WEIGHTS_SHA256 is set but $DFLASH_MODEL_HOST_PATH/model.safetensors does not exist (multi-shard drafts are not sha-pinned)"
    echo "glm53 pair launcher: hashing draft weights (~1.2 GB, a few seconds)..." >&2
    draft_sha=$(sha256sum "$DFLASH_MODEL_HOST_PATH/model.safetensors" | cut -d' ' -f1)
    [ "$draft_sha" = "$DFLASH_WEIGHTS_SHA256" ] \
      || die "draft weights sha256 mismatch: got $draft_sha, expected $DFLASH_WEIGHTS_SHA256 -- not the qualified checkpoint revision"
  fi
fi

for name in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_CUDAGRAPH_CAPTURE_SIZE; do
  require_positive_integer "$name"
done
case "$NUM_SPECULATIVE_TOKENS" in
  ''|*[!0-9]*) die "NUM_SPECULATIVE_TOKENS must be a non-negative integer: $NUM_SPECULATIVE_TOKENS" ;;
esac
if [ "$MAX_MODEL_LEN" != auto ]; then
  require_positive_integer MAX_MODEL_LEN
else
  warn "MAX_MODEL_LEN=auto resolves to the checkpoint maximum (1M for GLM-5.3); on a Spark pair the 1M profile can starve the KV pool and auto-fit the context below the attention page size -- set an explicit value (qualified: 262144)"
fi

case "$LANGUAGE_MODEL_ONLY" in
  0|1) : ;;
  *) die "LANGUAGE_MODEL_ONLY must be 0 or 1: $LANGUAGE_MODEL_ONLY" ;;
esac
[ "$LANGUAGE_MODEL_ONLY" = 1 ] \
  || warn "LANGUAGE_MODEL_ONLY=0 profiles the vision encoder against a maximum-size video item and reserves the encoder cache; expect several GiB less KV"

require_port API_PORT
require_port MASTER_PORT
[ "$API_PORT" != "$MASTER_PORT" ] || die "API_PORT and MASTER_PORT must differ"

require_unit_fraction GPU_MEMORY_UTILIZATION

# Setting kv_cache_memory_bytes makes vLLM skip memory profiling entirely, and
# gpu_memory_utilization is then ignored. Leave it empty for one boot to have
# vLLM profile at GPU_MEMORY_UTILIZATION and log a suggested value to pin.
# Remember GLM-5.3-Flash is a hybrid: the aligned mamba state cache lives in
# the same reservation as the paged KV pool, so pin the number vLLM reports,
# not one carried over from an attention-only model.
if [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
  require_positive_integer KV_CACHE_MEMORY_BYTES
  if grep -qE '^[[:space:]]*GPU_MEMORY_UTILIZATION=' "$env_file"; then
    warn "KV_CACHE_MEMORY_BYTES is set; vLLM skips profiling and IGNORES GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
  fi
else
  # GB10 unified memory: a node reports ~121.7 GiB total with ~10.5 GiB held
  # by the OS, so vLLM refuses any utilization above roughly 0.913 before it
  # loads anything.
  awk -v v="$GPU_MEMORY_UTILIZATION" 'BEGIN{ exit !(v+0 > 0.90) }' \
    && warn "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION is at or past this node's free-memory ceiling (~0.913); vLLM refuses at startup when the request exceeds free memory"

  if command -v nvidia-smi >/dev/null 2>&1; then
    mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    case "$mem_total$mem_free" in
      ''|*[!0-9]*) ;;
      *)
        awk -v t="$mem_total" -v f="$mem_free" -v u="$GPU_MEMORY_UTILIZATION" \
          'BEGIN{ want = t*u; if (want > f) printf "glm53 pair launcher: warning: GPU_MEMORY_UTILIZATION=%s asks for %.1f GiB but only %.1f GiB is free (ceiling %.4f); vLLM will refuse at startup\n", u, want/1024, f/1024, f/t > "/dev/stderr" }'
        ;;
    esac
  fi
fi

for name in ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL TRUST_REMOTE_CODE \
  VLLM_ENABLE_PCIE_ALLREDUCE VLLM_B12X_MOE_FP4_FORCE_A16; do
  require_bool "$name"
done

case "$KV_CACHE_DTYPE" in
  fp8|auto) ;;
  fp8_ds_mla|nvfp4_ds_mla) warn "KV_CACHE_DTYPE=$KV_CACHE_DTYPE is an r26+ value (packed NVFP4 KV is nvfp4_ds_mla); older images reject it" ;;
  *) die "KV_CACHE_DTYPE must be fp8, auto, fp8_ds_mla, or nvfp4_ds_mla: $KV_CACHE_DTYPE" ;;
esac
case "$DFLASH_KV_CACHE_DTYPE" in
  fp8|auto) ;;
  *) die "DFLASH_KV_CACHE_DTYPE must be fp8 or auto: $DFLASH_KV_CACHE_DTYPE" ;;
esac
case "$CUDAGRAPH_MODE" in
  FULL|FULL_AND_PIECEWISE|PIECEWISE|NONE) ;;
  *) die "CUDAGRAPH_MODE must be FULL, FULL_AND_PIECEWISE, PIECEWISE, or NONE: $CUDAGRAPH_MODE" ;;
esac

# GB10 is sm_121; the RTX Pro 6000 recipes this derives from use sm_120a.
[ "$CUTE_DSL_ARCH" = sm_121a ] \
  || die "CUTE_DSL_ARCH must be sm_121a on DGX Spark (got $CUTE_DSL_ARCH); sm_120a is the RTX Pro 6000 value"

# The B12X PCIe allreduce is an intra-node PCIe P2P path. On a two-node pair
# every TP allreduce crosses the ConnectX-7 link through NCCL; the PCIe path
# has nothing to attach to and the image's baked-in default of 1 must be
# overridden to 0 in the env file.
[ "$VLLM_ENABLE_PCIE_ALLREDUCE" = 0 ] \
  || die "VLLM_ENABLE_PCIE_ALLREDUCE must be 0 on a two-node pair (TP allreduce goes over RoCE via NCCL)"

# VLLM_B12X_MOE_FP4_FORCE_A16=0 (FP4-activation MoE path) is the value in the
# qualified 2026-08-28 recipe and the engine default; =1 forces the older,
# more conservative W4A16 path. Both are accepted.
case "$VLLM_B12X_MOE_FP4_FORCE_A16" in
  0|1) : ;;
  *) die "VLLM_B12X_MOE_FP4_FORCE_A16 must be 0 or 1: $VLLM_B12X_MOE_FP4_FORCE_A16" ;;
esac
[ "$VLLM_B12X_MOE_FP4_FORCE_A16" = 0 ] \
  || warn "VLLM_B12X_MOE_FP4_FORCE_A16=1 uses the pre-2026-08-28 W4A16 path; the current qualified recipe runs 0"

# Linear kernel selection: the qualified recipe pins --linear-backend b12x so
# the target's quantized linears and a DFlash MXFP8 draft both stay on
# sparkinfer kernels instead of flashinfer auto-selection (which has an
# unresolved CUTLASS SM121 MMA guard on GB10).
case "$API_KEY" in *[[:space:]]*) die "API_KEY must not contain whitespace" ;; esac
require_bool ENABLE_FLASHINFER_AUTOTUNE
case "$MOE_BACKEND" in b12x|humming|auto) : ;; *) die "MOE_BACKEND must be b12x, humming, or auto: $MOE_BACKEND" ;; esac
case "$ATTENTION_BACKEND" in B12X|auto) : ;; *) die "ATTENTION_BACKEND must be B12X or auto: $ATTENTION_BACKEND" ;; esac
case "$LINEAR_BACKEND" in
  b12x) : ;;
  auto) warn "LINEAR_BACKEND=auto may auto-select flashinfer linear kernels that are not qualified on sm_121" ;;
  *)    warn "LINEAR_BACKEND=$LINEAR_BACKEND is off the qualified recipe (b12x)" ;;
esac

# DFlash2 drafts a fixed eight-token block: one verified token plus seven
# draft tokens. Other values are legal but off the trained block size.
if [ "$SPECULATOR" = dflash2 ] && [ "$NUM_SPECULATIVE_TOKENS" != 7 ]; then
  warn "DFlash2 is trained for 7 draft tokens (8-token block); NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS is off the trained configuration"
fi

# One decode step must fit in a batch: max_num_seqs * (spec tokens + 1).
decode_batch=$(( MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1) ))
if [ "$decode_batch" -gt "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]; then
  warn "a full decode step is $decode_batch tokens (MAX_NUM_SEQS x (NUM_SPECULATIVE_TOKENS+1)) but MAX_CUDAGRAPH_CAPTURE_SIZE=$MAX_CUDAGRAPH_CAPTURE_SIZE; the largest batches fall out of cudagraph replay"
fi
if [ "$MAX_NUM_BATCHED_TOKENS" -lt "$decode_batch" ]; then
  die "MAX_NUM_BATCHED_TOKENS ($MAX_NUM_BATCHED_TOKENS) is below one speculative decode step ($decode_batch)"
fi

# ------------------------------------------------------------ fabric checks

[ "$NCCL_SOCKET_IFNAME" = "$GLOO_SOCKET_IFNAME" ] \
  || die "NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME must match on a pair"
[ "$NCCL_NET" = IB ] || die "NCCL_NET must be IB"
[ "$NCCL_NET_PLUGIN" = none ] || die "NCCL_NET_PLUGIN must be none"
# The image bakes NCCL_IB_DISABLE=1 for single-node PCIe boxes; the env file
# must override it back to 0 or the pair silently falls back to TCP sockets.
[ "$NCCL_IB_DISABLE" = 0 ] || die "NCCL_IB_DISABLE must be 0 (the serving image bakes 1 for single-node use; override it in the env file)"
# Two validated fabric profiles:
#   single-rail (default): MERGE_NICS=0, SUBNET_AWARE_ROUTING=0, one HCA
#   dual-rail (Luke's TP2 Spark launcher): MERGE_NICS=1,
#     SUBNET_AWARE_ROUTING=1, both rails listed in NCCL_IB_HCA -- NCCL
#     merges the two direct RoCE links for ~2x cross-node bandwidth.
# The two knobs must move together; a mixed setting is a misconfiguration.
if [ "$NCCL_IB_MERGE_NICS" = 1 ]; then
  [ "$NCCL_IB_SUBNET_AWARE_ROUTING" = 1 ] \
    || die "dual-rail profile requires NCCL_IB_SUBNET_AWARE_ROUTING=1 with NCCL_IB_MERGE_NICS=1"
  case "$NCCL_IB_HCA" in
    *,*) : ;;
    *) warn "NCCL_IB_MERGE_NICS=1 with a single HCA in NCCL_IB_HCA -- dual-rail wants both rails listed (e.g. rocep1s0f0,rocep1s0f1)" ;;
  esac
elif [ "$NCCL_IB_SUBNET_AWARE_ROUTING" != 0 ]; then
  warn "NCCL_IB_SUBNET_AWARE_ROUTING=$NCCL_IB_SUBNET_AWARE_ROUTING with NCCL_IB_MERGE_NICS=0: off the validated single-rail profile (0/0); upstream's TP2 Spark launcher runs 1/0, so this is allowed but unmeasured here"
fi
[ "$NCCL_PROTO" = LL,LL128,Simple ] || die "NCCL_PROTO must be LL,LL128,Simple"
[ "$NCCL_P2P_LEVEL" = SYS ] || die "NCCL_P2P_LEVEL must be SYS"
[ "$NCCL_CROSS_NIC" = 1 ] || die "NCCL_CROSS_NIC must be 1"
[ "$NCCL_CUMEM_ENABLE" = 0 ] || die "NCCL_CUMEM_ENABLE must be 0"
[ "$NCCL_IGNORE_CPU_AFFINITY" = 1 ] || die "NCCL_IGNORE_CPU_AFFINITY must be 1"

case ":$LD_PRELOAD:" in
  *":$VLLM_NCCL_SO_PATH:"*) ;;
  *) die "LD_PRELOAD must include VLLM_NCCL_SO_PATH ($VLLM_NCCL_SO_PATH)" ;;
esac

[ "$NODE_RANK" != 0 ] || [ "$MASTER_ADDR" = "$VLLM_HOST_IP" ] \
  || die "rank-0 MASTER_ADDR must equal rank-0 VLLM_HOST_IP"
[ "$NODE_RANK" != 1 ] || [ "$MASTER_ADDR" != "$VLLM_HOST_IP" ] \
  || die "rank-1 VLLM_HOST_IP must differ from MASTER_ADDR"

# Preflight the memory vLLM will demand at startup. On Grace unified memory,
# CUDA free-memory reporting tracks MemFree, and page cache left behind by a
# previous 184 GB model load (or a stale container) counts AGAINST it --
# vLLM's own check then fails with "Free memory ... is less than desired GPU
# memory utilization" before loading anything. Catch and name the culprit
# here instead.
: "${MEM_PREFLIGHT:=die}"   # die | warn | off
case "$MEM_PREFLIGHT" in die|warn|off) : ;; *) die "MEM_PREFLIGHT must be die, warn, or off" ;; esac
if [ "$MEM_PREFLIGHT" != off ] && [ -r /proc/meminfo ] && [ -z "$KV_CACHE_MEMORY_BYTES" ]; then
  mem_total_kib=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
  mem_free_kib=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
  mem_avail_kib=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  required_kib=$(awk -v t="$mem_total_kib" -v u="$GPU_MEMORY_UTILIZATION" 'BEGIN{printf "%d", t*u}')
  if [ "$mem_free_kib" -lt "$required_kib" ]; then
    deficit_gib=$(awk -v r="$required_kib" -v f="$mem_free_kib" 'BEGIN{printf "%.1f", (r-f)/1048576}')
    stale="$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c '^glm53-flash-r' || true)"
    if [ "${stale:-0}" -gt 0 ]; then
      die "only $(awk -v f="$mem_free_kib" 'BEGIN{printf "%.1f", f/1048576}') GiB free but utilization $GPU_MEMORY_UTILIZATION needs ~$(awk -v r="$required_kib" 'BEGIN{printf "%.1f", r/1048576}') GiB (short $deficit_gib GiB) -- a glm53-flash container is already running on this host; --down it first"
    elif [ "$mem_avail_kib" -ge "$required_kib" ]; then
      cache_gib=$(awk -v a="$mem_avail_kib" -v f="$mem_free_kib" 'BEGIN{printf "%.1f", (a-f)/1048576}')
      msg="MemFree=$(awk -v f="$mem_free_kib" 'BEGIN{printf "%.1f", f/1048576}') GiB is $deficit_gib GiB short of utilization $GPU_MEMORY_UTILIZATION, but MemAvailable=$(awk -v a="$mem_avail_kib" 'BEGIN{printf "%.1f", a/1048576}') GiB suffices: ~$cache_gib GiB of reclaimable page cache (a previous model load) is counting against CUDA free memory. Dashboards show total-minus-available, so the node LOOKS idle -- but vLLM's own startup gate reads CUDA-free (~MemFree) and will refuse, exactly as it did at 105.19/121.69 GiB previously. Reclaim: sync && echo 3 | sudo tee /proc/sys/vm/drop_caches -- then relaunch (MEM_PREFLIGHT=warn to proceed anyway)"
      if [ "$MEM_PREFLIGHT" = warn ]; then warn "$msg"; else die "$msg"; fi
    else
      die "this host has only $(awk -v a="$mem_avail_kib" 'BEGIN{printf "%.1f", a/1048576}') GiB available; utilization $GPU_MEMORY_UTILIZATION needs ~$(awk -v r="$required_kib" 'BEGIN{printf "%.1f", r/1048576}') GiB. Find the consumer (docker ps, top) or lower GPU_MEMORY_UTILIZATION"
    fi
  fi
fi

# MASTER_ADDR must sit on a directly connected subnet (the CX7 link or a
# local address), never behind a gateway. A one-digit typo like
# 198.168.x.x routes to the internet and the torch.distributed rendezvous
# hangs forever with no error on either node -- catch it here instead.
if command -v ip >/dev/null 2>&1; then
  master_route="$(ip route get "$MASTER_ADDR" 2>/dev/null | head -1)"
  case "$master_route" in
    "")        die "MASTER_ADDR=$MASTER_ADDR has no route from this host -- typo?" ;;
    *" via "*) die "MASTER_ADDR=$MASTER_ADDR is not on a directly connected subnet (route: $master_route) -- almost certainly a typo; it must be rank 0's address on the CX7 link" ;;
  esac
fi

case "$NCCL_IB_HCA" in
  *,*) [ "$NCCL_IB_MERGE_NICS" = 1 ] \
         || die "multiple RoCE devices in NCCL_IB_HCA require the dual-rail profile (NCCL_IB_MERGE_NICS=1, SUBNET_AWARE_ROUTING=1)" ;;
esac
case "$NCCL_IB_GID_INDEX" in
  ''|*[!0-9]*) die "NCCL_IB_GID_INDEX must be a decimal integer" ;;
esac

# ------------------------------------------------------- host / image checks

# Running the wrong rank's env file on a host is easy to do and hard to
# diagnose: both ranks would fight over MASTER_PORT and API_PORT under
# --network host. VLLM_HOST_IP must be an address this machine actually holds.
if command -v ip >/dev/null 2>&1; then
  if ! ip -4 -o addr show 2>/dev/null | grep -qw "$VLLM_HOST_IP"; then
    die "VLLM_HOST_IP ($VLLM_HOST_IP) is not an IPv4 address on this host; wrong rank's env file?"
  fi
fi

# --network host means an occupied port is an immediate bind failure at exec.
port_in_use() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"
  else
    return 1
  fi
}
if [ "$NODE_RANK" = 0 ]; then
  ! port_in_use "$API_PORT" \
    || die "API_PORT $API_PORT is already listening on this host; --down the old container or pick another port"
  ! port_in_use "$MASTER_PORT" \
    || die "MASTER_PORT $MASTER_PORT is already listening on this host"
fi

# The RDMA device node must exist or `--device /dev/infiniband` fails at
# docker run with an error that names the path but not the cause.
[ -e /dev/infiniband ] \
  || die "/dev/infiniband does not exist; the RDMA stack is not up (check: lsmod | grep mlx5_ib)"

# --ipc host makes the HOST's /dev/shm authoritative; the container --shm-size
# declaration is inert. vLLM's shm_broadcast stalls when this is small.
if [ -d /dev/shm ]; then
  shm_mb=$(df -Pm /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
  case "$shm_mb" in
    ''|*[!0-9]*) ;;
    *) [ "$shm_mb" -ge 8192 ] \
         || warn "/dev/shm is ${shm_mb} MiB; --ipc host makes this the real limit (--shm-size is inert) and vLLM's shm_broadcast can stall below ~8 GiB" ;;
  esac
fi

# LD_PRELOAD is the single most fragile part: if either path is absent from the
# image, EVERY process in the container dies at exec with an opaque dynamic
# loader error, or later with `undefined symbol: cuTensorMapEncodeTiled` when
# the CUDA compat shim is the missing one. Verify inside the image itself.
if [ "${SKIP_IMAGE_PRELOAD_CHECK:-0}" != 1 ] \
   && command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1 \
   && "$CONTAINER_RUNTIME" image inspect "$SERVING_IMAGE" >/dev/null 2>&1; then
  preload_missing=""
  old_ifs=$IFS; IFS=:
  for lib in $LD_PRELOAD; do
    [ -n "$lib" ] || continue
    "$CONTAINER_RUNTIME" run --rm --entrypoint test "$SERVING_IMAGE" -f "$lib" \
      >/dev/null 2>&1 || preload_missing="$preload_missing $lib"
  done
  IFS=$old_ifs
  [ -z "$preload_missing" ] \
    || die "LD_PRELOAD names path(s) absent from the image:$preload_missing -- every process in the container would fail at exec"
fi

# ----------------------------------------------------------- launch command

container_name="$mgmt_container"
model_container_path=/models/glm-5.3-flash-nvfp4
dflash_container_path=/models/glm-5.3-flash-dflash2

# Split cache pages are the operator's call and must be explicit in the env
# file (they reach the container via --env-file). --check only verifies the
# combination against what was measured on pin 54f6e982 (2026-09-02):
#   dflash2 without them -> boot fails ("parent-page stride must be an exact
#     number of C4 pages"); 4096/4096 boots (731,620 tokens, 12 GiB pin).
#   mtp with 512 -> pool ~10x smaller (13.74 GiB per 262k request); with
#     4096 -> no measured gain over the coupled default.
if [ "$SPECULATOR" = dflash2 ]; then
  if [ -z "${VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE-}" ] || [ -z "${VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE-}" ]; then
    warn "SPECULATOR=dflash2 without split cache pages: at pin 54f6e982 this failed at boot. Add to BOTH env files: VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=4096 and VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=4096"
  fi
elif [ -n "${VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE-}" ] && [ "${VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE}" -lt 2048 ] 2>/dev/null; then
  warn "VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=$VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE under SPECULATOR=$SPECULATOR: on pins before #669 (2026-09-06), 512 inflated every slot to the recurrent page (13.74 GiB per 262k request). Pins >= db7a65e2 handle small targets (upstream runs 512/512; El8 2048/256); older pins want 2048 or the coupled default"
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
case "$INSTANTTENSOR_COPY" in
  auto) : ;;
  0|1) warn "INSTANTTENSOR_COPY=$INSTANTTENSOR_COPY passes instanttensor_copy to the loader; vllm pins >= 6575b5ac (2026-09-05) removed that option (launchers moved to --load-format fastsafetensors)" ;;&
  0) extra_args+=(--model-loader-extra-config '{"instanttensor_copy":false}') ;;
  1) extra_args+=(--model-loader-extra-config '{"instanttensor_copy":true}') ;;
  *) die "INSTANTTENSOR_COPY must be auto, 0, or 1" ;;
esac
[ -z "$COMPILATION_LEVEL" ] || extra_args+=("-O$COMPILATION_LEVEL")
[ -z "$KDA_PREFILL_BACKEND" ] || extra_args+=(--kda-prefill-backend "$KDA_PREFILL_BACKEND")
[ -z "$GDN_DECODE_KERNEL" ] || extra_args+=(--gdn-decode-kernel "$GDN_DECODE_KERNEL")
[ -z "$FAIRNESS_ENGINE" ] || extra_args+=(--fairness-engine "$FAIRNESS_ENGINE")
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
[ -z "$PREFILL_COMPUTE_SHARE" ] || extra_args+=(--prefill-compute-share "$PREFILL_COMPUTE_SHARE")
[ -z "$MM_PROCESSOR_CACHE_GB" ] || extra_args+=(--mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB")
[ -z "$MM_ENCODER_TP_MODE" ] || extra_args+=(--mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")
[ "$GENERATION_CONFIG" = auto ] || extra_args+=(--generation-config "$GENERATION_CONFIG")
chat_template_container_path=/models/chat_template.jinja
[ -z "$CHAT_TEMPLATE_HOST_PATH" ] || extra_args+=(--chat-template "$chat_template_container_path")

quantization_args=()
if [ "$QUANTIZATION" != auto ]; then
  quantization_args=(--quantization "$QUANTIZATION")
fi

lm_only_args=()
if [ "$LANGUAGE_MODEL_ONLY" = 1 ]; then
  lm_only_args=(--language-model-only)
else
  # Cap per-prompt multimodal items instead of the 999-per-modality default.
  # The JSON is built HERE, not read from the env file: the env file is
  # sourced by this script, and shell sourcing strips double quotes from
  # unquoted values -- {"image":4} arrives as {image:4} and vllm's
  # json.loads rejects it. Integer envs have no such failure mode.
  require_bool LIMIT_MM
  if [ "$LIMIT_MM" = 1 ]; then
    # 0 is legitimate: it disables that modality (MM_VIDEOS=0 drops the
    # max-size video profile item, most of the vision memory reservation).
    require_nonnegative_integer MM_IMAGES
    require_nonnegative_integer MM_VIDEOS
    [ "$((MM_IMAGES + MM_VIDEOS))" -gt 0 ] \
      || warn "MM_IMAGES=0 and MM_VIDEOS=0 disable every modality; LANGUAGE_MODEL_ONLY=1 is the clearer way to say that"
    lm_only_args=(--limit-mm-per-prompt "$(printf '{"image":%d,"video":%d}' "$MM_IMAGES" "$MM_VIDEOS")")
  else
    # No cap: the engine default is 999 per modality per prompt (the whole
    # chat history counts). Every image still costs context tokens; with
    # profiling (no KV pin) the profiler sizes its dummy prompt from this
    # limit, so leave the pin set when running uncapped.
    lm_only_args=()
    [ -n "$KV_CACHE_MEMORY_BYTES" ] \
      || warn "LIMIT_MM=0 without KV_CACHE_MEMORY_BYTES: memory profiling will size its dummy prompt at the engine's 999-per-modality default"
  fi
fi

speculative_args=()
if [ "$SPECULATOR" != none ] && [ "$NUM_SPECULATIVE_TOKENS" -gt 0 ]; then
  case "$SPECULATOR" in
    mtp)
      # The MTP head is BF16 in this checkpoint; humming carries its MoE and
      # B12X its attention, per the canonical GLM-5.3 recipe.
      adaptive_fields=
      if [ "$ADAPTIVE_SPECULATIVE_TOKENS" = 1 ]; then
        adaptive_fields=$(printf ',"adaptive_speculative_tokens_window":%s,"adaptive_speculative_tokens_initial":%s' \
          "$ADAPTIVE_SPECULATIVE_TOKENS_WINDOW" "$ADAPTIVE_SPECULATIVE_TOKENS_INITIAL")
      fi
      speculative_config=$(printf \
        '{"method":"mtp","num_speculative_tokens":%s,"moe_backend":"%s","attention_backend":"%s"%s}' \
        "$NUM_SPECULATIVE_TOKENS" "${MTP_MOE_BACKEND:-humming}" "${MTP_ATTENTION_BACKEND:-B12X}" "$adaptive_fields")
      ;;
    dflash2)
      # Works for both draft checkpoints: BF16 (incoai/GLM-5.3-Flash-DFlash2)
      # and MXFP8 (local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8) -- the
      # quantization is read from the checkpoint config; point
      # DFLASH_MODEL_HOST_PATH at whichever is downloaded. kv_cache_dtype
      # auto (BF16) is the qualified value for both.
      speculative_config=$(printf \
        '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"kv_cache_dtype":"%s"}' \
        "$dflash_container_path" "$NUM_SPECULATIVE_TOKENS" "$DFLASH_KV_CACHE_DTYPE")
      ;;
  esac
  speculative_args=(--speculative-config "$speculative_config")
fi

compilation_config=$(printf '{"cudagraph_mode":"%s","custom_ops":["all"]}' "$CUDAGRAPH_MODE")

case "$CONTAINER_RUNTIME" in
  podman) gpu_args=(--device nvidia.com/gpu=all --security-opt label=disable) ;;
  *)      gpu_args=(--gpus all) ;;
esac

mount_args=(
  -v "$MODEL_HOST_PATH:$model_container_path:ro"
  -v "$CACHE_HOST_PATH:/cache"
)
if [ "$SPECULATOR" = dflash2 ]; then
  mount_args+=(-v "$DFLASH_MODEL_HOST_PATH:$dflash_container_path:ro")
fi
if [ -n "$CHAT_TEMPLATE_HOST_PATH" ]; then
  mount_args+=(-v "$CHAT_TEMPLATE_HOST_PATH:$chat_template_container_path:ro")
fi
if [ -n "$TORCH_PROFILE_HOST_DIR" ]; then
  mkdir -p "$TORCH_PROFILE_HOST_DIR"
  mount_args+=(-v "$TORCH_PROFILE_HOST_DIR:/profiles")
fi

command=(
  "$CONTAINER_RUNTIME" run -d
  --name "$container_name"
  --pull never
  --network host
  --ipc host
  --shm-size "$SHM_SIZE"
  "${gpu_args[@]}"
  --ulimit memlock=-1:-1
  ${CONTAINER_MEMORY_GB:+--memory ${CONTAINER_MEMORY_GB}g --memory-swap $((${CONTAINER_MEMORY_GB:-0}+4))g}
  --device /dev/infiniband
  "${mount_args[@]}"
  --env-file "$env_file"
  # The image bakes VLLM_PCIE_ALLREDUCE_BACKEND=cpp (the pre-rename value).
  # The pinned vLLM validates this var with choices ['b12x'] and its eager
  # env cache evaluates EVERY var at worker start, so the stale name crashes
  # workers even though VLLM_ENABLE_PCIE_ALLREDUCE=0 means the PCIe path is
  # never used. Explicit -e beats both the env file and the baked ENV.
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
  ${TORCH_PROFILE_HOST_DIR:+-e VLLM_TORCH_PROFILER_DIR=/profiles}
  # TP=2 InstantTensor staging bounds (canonical launcher parity; caps the
  # checkpoint-loading memory peak on the new pins, ignored by older images).
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
  --decode-context-parallel-size 1

  # --- memory -------------------------------------------------------------
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --block-size "$BLOCK_SIZE"

  # --- model / kernels (canonical GLM-5.3-Flash-NVFP4 recipe) --------------
  --dtype bfloat16
  "${quantization_args[@]}"
  --attention-backend "$ATTENTION_BACKEND"
  --moe-backend "$MOE_BACKEND"
  --linear-backend "$LINEAR_BACKEND"
  "${lm_only_args[@]}"
  --mamba-cache-mode align
  --load-format "$LOAD_FORMAT"
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

# Guarded with `if` rather than `&&` so a disabled toggle does not trip `set -e`.
if [ "$TRUST_REMOTE_CODE" = 1 ]; then command+=(--trust-remote-code); fi
if [ "$ENABLE_PREFIX_CACHING" = 1 ]; then command+=(--enable-prefix-caching); fi
if [ "$ENABLE_CHUNKED_PREFILL" = 1 ]; then command+=(--enable-chunked-prefill); fi
if [ "$ENABLE_FLASHINFER_AUTOTUNE" = 1 ]; then command+=(--enable-flashinfer-autotune); else command+=(--no-enable-flashinfer-autotune); fi
if [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
  command+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi

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
printf '  runtime:                 %s\n' "$CONTAINER_RUNTIME"
printf '  model:                   %s\n' "$MODEL_HOST_PATH"
if [ "$SPECULATOR" = dflash2 ]; then
  printf '  draft model:             %s\n' "$DFLASH_MODEL_HOST_PATH"
fi
printf '  cache:                   %s\n' "$CACHE_HOST_PATH"
printf '  MAX_MODEL_LEN:           %s\n' "$MAX_MODEL_LEN"
printf '  MAX_NUM_SEQS:            %s\n' "$MAX_NUM_SEQS"
printf '  MAX_NUM_BATCHED_TOKENS:  %s\n' "$MAX_NUM_BATCHED_TOKENS"
printf '  SPECULATOR:              %s (%s draft tokens)\n' \
  "$SPECULATOR" "$NUM_SPECULATIVE_TOKENS"
printf '  KV_CACHE_MEMORY_BYTES:   %s (%s)\n' \
  "${KV_CACHE_MEMORY_BYTES:-profiled at $GPU_MEMORY_UTILIZATION}" "$KV_CACHE_DTYPE"
if [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
  printf '  GPU_MEMORY_UTILIZATION:  %s (IGNORED - bytes are pinned)\n' "$GPU_MEMORY_UTILIZATION"
else
  printf '  GPU_MEMORY_UTILIZATION:  %s\n' "$GPU_MEMORY_UTILIZATION"
fi
printf '  CUDAGRAPH_MODE:          %s (capture %s)\n' \
  "$CUDAGRAPH_MODE" "$MAX_CUDAGRAPH_CAPTURE_SIZE"
printf '  LOAD_FORMAT:             %s\n' "$LOAD_FORMAT"
printf '  API_KEY:                 %s\n' "$([ -n "$API_KEY" ] && echo 'set (Bearer required)' || echo 'none (open port)')"
printf '  command:'
printf ' %q' "${command[@]}"
printf '\n'

[ "$mode" = --run ] || exit 0

# --------------------------------------------------------------------- run

command -v "$CONTAINER_RUNTIME" >/dev/null 2>&1 \
  || die "$CONTAINER_RUNTIME is unavailable"
"$CONTAINER_RUNTIME" image inspect "$SERVING_IMAGE" >/dev/null 2>&1 \
  || die "pinned image is not present; build or load it before launching: $SERVING_IMAGE"
if "$CONTAINER_RUNTIME" container inspect "$container_name" >/dev/null 2>&1; then
  die "container already exists; remove it intentionally before relaunch: $container_name"
fi

# --fresh: started detached on purpose, so do NOT exec away the shell -- the
# launcher follows the logs after the container is up. Ctrl-C detaches; the
# container keeps running.
if [ "$fresh_follow" = 1 ]; then
  "${command[@]}"
  printf '%s started; following logs (Ctrl-C detaches, the container keeps running)\n' "$container_name"
  exec "$CONTAINER_RUNTIME" logs -f "$container_name"
else
  exec "${command[@]}"
fi
