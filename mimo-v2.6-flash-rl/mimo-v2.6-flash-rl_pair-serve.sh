#!/usr/bin/env bash
# mimo-v2.6-flash-rl_pair-serve.sh
#
# Validate or start one rank of a two-node DGX Spark pair (TP=2 over the
# direct ConnectX-7 RoCE link) serving XiaomiMiMo/MiMo-V2.6-Flash-RL under
# vLLM (karmic-kraken @ e77be225 + b12x @ 8a99d639 -- the kk-beta-cu132
# profile). Model: MiMoV2ForCausalLM -- 48 layers, hidden 4096, head_dim 192
# full attention alternating with SWA layers (hybrid_layer_pattern), MXFP4
# routed experts + block-FP8 projections, image/video encoders, audio via the
# MiMoV2OmniForCausalLM override (HF_OVERRIDES knob; not the groundwork
# default). NO mamba/GDN/recurrent state: pure-attention hybrid.
#
# kk-beta-cu132 (2026-09-23): the b12x loader is REQUIRED for this checkpoint
# (MiMo checkpoint + auxiliary-state loading lives in it -- de16d74135,
# 308fb7204f; the catalog's load_format: b12x). The loader reads weights
# through an O_DIRECT io_uring ring (b12x 1ec67ef6), which needs
# io_uring_setup/enter/register in the container seccomp: the launcher passes
# the SHARED ../seccomp-io-uring.json at the deployments root (Moby default +
# the three syscalls = vLLM scripts/seccomp/spark-io-uring.json) and probes
# io_uring inside the image at --check (shared block before command=()).
#
# GROUNDWORK STATUS: the catalog qualifies MiMo at TP4 on SM120 (sm_120a)
# ONLY; GB10/sm_121a is unqualified territory -- expect kernel surprises and
# treat every value below as the vendor contract (lil.yaml + Luke's
# serve-mimo26-flash.sh), not a measured one. Speculation: none (qualified);
# the in-checkpoint MTP head is the first A/B. Set max_model_len 262144 for
# the first boots; the checkpoint's 1,048,576 ceiling is the post-bring-up
# target.
#
# Usage:
#     ./mimo-v2.6-flash-rl_pair-serve.sh --check   rank-0.env
#     ./mimo-v2.6-flash-rl_pair-serve.sh --run     rank-0.env
#     ./mimo-v2.6-flash-rl_pair-serve.sh --restart rank-0.env
#     ./mimo-v2.6-flash-rl_pair-serve.sh --down    rank-0.env
#     ./mimo-v2.6-flash-rl_pair-serve.sh --logs    rank-0.env
#     ./mimo-v2.6-flash-rl_pair-serve.sh --verify  rank-0.env
#     ./mimo-v2.6-flash-rl_pair-serve.sh --status  rank-0.env
#     ./mimo-v2.6-flash-rl_pair-serve.sh --clear   rank-0.env
#     ./mimo-v2.6-flash-rl_pair-serve.sh --fresh   rank-0.env
#
# Anything after ENV_FILE is appended verbatim to the vllm serve argv.
#
# Start rank 1 (headless, waits for rank 0) before rank 0.

set -euo pipefail

# Directory of this script: the shared seccomp profile resolves from here.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<'EOF'
usage: mimo-v2.6-flash-rl_pair-serve.sh [MODE] ENV_FILE [extra vllm args...]

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
  echo "mimo pair launcher: $*" >&2
  exit 20
}

warn() {
  echo "mimo pair launcher: warning: $*" >&2
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

# --env-file delivers KEY= as an EMPTY STRING, never as "unset". The
# launcher's own keys treat empty as "omit the flag"; engine-read variables
# do not (VLLM_PREFIX_CACHE_RETENTION_INTERVAL= crashed argparse with int('')
# on 2026-09-11). Refuse an empty value for any key the launcher does not
# consume. Models extend the allow-list via MODEL_EMPTY_OK_KEYS
# (space-separated).
LAUNCHER_EMPTY_OK_KEYS="API_KEY NUM_SPECULATIVE_TOKENS KV_CACHE_MEMORY_BYTES CONTAINER_MEMORY_GB CONTAINER_NAME_SUFFIX PREFILL_SCHEDULE_INTERVAL FAIRNESS_ENGINE PREFILL_COMPUTE_SHARE PREFILL_COMPUTE_HALF_LIFE MAX_PARALLEL_PREFILLS SECCOMP_PROFILE CUSTOM_OPS ASYNC_SCHEDULING PREFIX_RETENTION_INTERVAL COMPILATION_LEVEL SAMPLING_TEMPERATURE SAMPLING_TOP_P SAMPLING_TOP_K SAMPLING_MIN_P SAMPLING_REPETITION_PENALTY MM_PROCESSOR_CACHE_GB MM_ENCODER_TP_MODE CHAT_TEMPLATE_HOST_PATH TORCH_PROFILE_HOST_DIR VLLM_PLUGINS"
MODEL_EMPTY_OK_KEYS="B12X_AUTOTUNE"
empty_bad=""
for k in $(grep -Eo '^[[:space:]]*[A-Z][A-Z0-9_]*=[[:space:]]*$' "$env_file" | tr -d ' ='); do
  case " ${LAUNCHER_EMPTY_OK_KEYS} ${MODEL_EMPTY_OK_KEYS:-} " in
    *" $k "*) ;;
    *) empty_bad="$empty_bad $k" ;;
  esac
done
[ -z "$empty_bad" ] || die "empty value(s) not allowed for engine-read key(s):$empty_bad -- delete the line instead (empty is a VALUE, not unset)"

# ------------------------------------------------- container management modes
# These need only the rank and runtime, so they run before full validation.

: "${CONTAINER_RUNTIME:=docker}"
: "${CONTAINER_NAME_SUFFIX:=}"
mgmt_container="mimo-v26-flash-r${NODE_RANK:-?}${CONTAINER_NAME_SUFFIX}"

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
    # Markers that matter on the pair: the MiMo model class, the b12x
    # attention/linear/MoE lines, the KV/cudagraph lines. Absence of a marker
    # you expect means the wrong path was taken.
    "$CONTAINER_RUNTIME" logs "$mgmt_container" 2>&1 | grep -E \
      'speculative_config|MiMo|Mimo|b12x|B12X|B12x|indexer|Indexer|BLHNC|Loading (safetensors|weights)|Using .* all-reduce backends|RoCEnante|attention block size|Available KV cache memory|GPU KV cache size|Maximum concurrency|Graph capturing finished|cudagraph_mode=|Application startup complete' \
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
# Serving flags mirror serve-mimo26-flash.sh (Luke's canonical MiMo recipe at
# the pin) -- with the two-node topology and DGX Spark fabric handling carried
# over from the validated GLM pair launcher. Values are the VENDOR CONTRACT
# (lil.yaml + Luke's launcher), not pair measurements.

: "${SERVED_MODEL_NAME:=XiaomiMiMo/MiMo-V2.6-Flash-RL}"
: "${SPECULATOR:=none}"                # none | mtp (3 in-checkpoint MTP layers) | dflash (the dflash/ draft shipped in the Flash-RL snapshot: 5 SWA layers, block 8)
: "${GPU_MEMORY_UTILIZATION:=0.90}"   # pair default; GB10 cu132 ceiling ~0.913; catalog's 0.98 is the RTX TP4 value
: "${KV_CACHE_MEMORY_BYTES:=}"         # empty = let vLLM profile and choose (MiMo pool math never measured anywhere)
: "${KV_CACHE_DTYPE:=bfloat16}"        # catalog-qualified for MiMo's 192-dim attention; fp8 UNVERIFIED on this model (A/B later)
: "${QUANTIZATION:=auto}"              # auto = omit the flag. The checkpoint is quant_method fp8 (store_dtype mxfp4), NOT ModelOpt: modelopt_mixed fails vLLM's quant-method check. KK serve-mimo26-flash.sh passes no --quantization
: "${PREFILL_SCHEDULE_INTERVAL:=}"     # empty = engine default; the GLM pair's 8 is an unmeasured import here
: "${COMPILATION_LEVEL:=}"             # empty = no -O flag; 0-3 passes -O<N> (torch.compile level) -- A/B only
: "${GENERATION_CONFIG:=vllm}"         # catalog: vllm (MiMo's generation_config is not authoritative); override-generation-config carries 1.0/0.95 (Luke's launcher)
: "${SAMPLING_TEMPERATURE:=}"          # empty = launcher's override (1.0/0.95 below); per-request fields still win
: "${SAMPLING_TOP_P:=}"
: "${SAMPLING_TOP_K:=}"
: "${SAMPLING_MIN_P:=}"
: "${SAMPLING_REPETITION_PENALTY:=}"
: "${FAIRNESS_ENGINE:=}"               # RETIRED on karmic-kraken (no --fairness-engine); must stay empty
: "${PREFILL_COMPUTE_SHARE:=}"         # empty = off; fraction in (0,1) or auto; requires PREFILL_SCHEDULE_INTERVAL=1. The third-party attachment runs 0.8 (TP4) -- unmeasured here
: "${PREFILL_COMPUTE_HALF_LIFE:=}"     # only with PREFILL_COMPUTE_SHARE=auto; seconds, or smooth (2 s) | responsive (0.5 s)
: "${MAX_PARALLEL_PREFILLS:=}"         # empty = engine default (1); positive integer or auto
: "${BLOCK_SIZE:=128}"                 # catalog-qualified geometry for MiMo's 192/128 attention (GLM's 256 is NOT this model's value)
: "${LIMIT_MM:=1}"                     # 1 = pass --limit-mm-per-prompt from MM_IMAGES/MM_VIDEOS; 0 = omit it (engine default: 999 per modality)
: "${MM_PROCESSOR_CACHE_GB:=0}"        # 0 disables (catalog multimodal: processor_cache_gb 0); on unified memory the 4 GiB default is KV pool
: "${MM_ENCODER_TP_MODE:=weights}"     # catalog-qualified (MiMo carries image+video+AUDIO encoders; data would replicate all towers per rank)
: "${CHAT_TEMPLATE_HOST_PATH:=}"       # optional custom .jinja
: "${TORCH_PROFILE_HOST_DIR:=}"        # set to a host dir to enable /start_profile + /stop_profile
: "${CONTAINER_MEMORY_GB:=}"           # cgroup cap so a runaway load cannot OOM the host
: "${ASYNC_SCHEDULING:=}"              # 1 = --async-scheduling (NOT in the MiMo catalog contract; A/B)
: "${PREFIX_RETENTION_INTERVAL:=}"     # --prefix-cache-retention-interval N (SWA checkpoints; NOT catalog-qualified for MiMo; block-size multiple)
: "${VLLM_KV_CACHE_LAYOUT:=}"          # empty = engine auto; BLHNC (block-outermost) is required ONLY when mixing draft+target page sizes (DFlash A/B; fc27214ba9)
# TP=2 InstantTensor staging bounds (inert under the b12x loader; only passed
# when instanttensor is selected).
: "${INSTANTTENSOR_BUFFER_SIZE:=1342177280}"
: "${INSTANTTENSOR_IO_DEPTH:=3}"
: "${INSTANTTENSOR_CONCURRENCY:=1}"
: "${INSTANTTENSOR_CHUNK_SIZE:=8388608}"
: "${INSTANTTENSOR_COPY:=auto}"
: "${API_KEY:=}"                       # empty = open port; set to require Authorization: Bearer <key> on rank 0
: "${ENABLE_FLASHINFER_AUTOTUNE:=0}"   # catalog kernels.flashinfer_autotune false; B12X owns attention/MoE/linear
: "${MOE_BACKEND:=b12x}"               # catalog contract: b12x | humming | auto
: "${ATTENTION_BACKEND:=B12X}"          # catalog contract: B12X | auto
: "${LINEAR_BACKEND:=b12x}"            # catalog contract: b12x
: "${LANGUAGE_MODEL_ONLY:=0}"          # 0 = image/video ON (the catalog serves the full multimodal checkpoint)
: "${MM_IMAGES:=8}"                    # per-prompt image cap when LIMIT_MM=1 (groundwork value)
: "${MM_VIDEOS:=0}"                    # per-prompt video cap when LIMIT_MM=1
: "${MAX_MODEL_LEN:=262144}"          # groundwork value; the checkpoint ceiling is 1,048,576 (raise after a profiled boot)
: "${MAX_NUM_SEQS:=4}"                 # catalog capacity value
: "${MAX_NUM_BATCHED_TOKENS:=4096}"   # catalog value
: "${MAX_CUDAGRAPH_CAPTURE_SIZE:=16}" # covers 4 seqs x (3+1) with headroom; catalog TP1 runs 64 at 16 seqs
: "${CUDAGRAPH_MODE:=FULL_AND_PIECEWISE}" # catalog compilation value
: "${LOAD_FORMAT:=b12x}"               # REQUIRED for MiMo (checkpoint + auxiliary-state loading lives in the b12x loader)
: "${ENABLE_PREFIX_CACHING:=1}"
: "${ENABLE_CHUNKED_PREFILL:=1}"
: "${TRUST_REMOTE_CODE:=1}"            # catalog: trust_remote_code true
: "${CONTAINER_RUNTIME:=docker}"
: "${CONTAINER_NAME_SUFFIX:=}"
: "${SERVING_IMAGE:=local/vllm:karmic-kraken-beta-cu132}"
: "${SHM_SIZE:=16g}"

# Accept MTP=<n> as an alias for NUM_SPECULATIVE_TOKENS under SPECULATOR=mtp,
# matching the single-node run commands this recipe was derived from.
if [ -n "${MTP-}" ] && [ -z "${NUM_SPECULATIVE_TOKENS-}" ]; then
  NUM_SPECULATIVE_TOKENS=$MTP
fi

: "${DFLASH_SUBDIR:=dflash}"            # draft location inside MODEL_HOST_PATH (the HF snapshot ships dflash/)
: "${DFLASH_KV_CACHE_DTYPE:=auto}"      # KK serve-mimo26-pro.sh: draft KV "auto" (BF16)
case "$SPECULATOR" in
  none)    : "${NUM_SPECULATIVE_TOKENS:=0}" ;;
  mtp)     : "${NUM_SPECULATIVE_TOKENS:=3}" ;;   # 3 = one trained MTP layer per step (num_nextn_predict_layers 3)
  dflash)  : "${NUM_SPECULATIVE_TOKENS:=7}" ;;   # dflash/config.json block_size 8 -> 7 draft tokens (KK Pro launcher default 7)
  *) die "SPECULATOR must be none, mtp, or dflash: $SPECULATOR" ;;
esac

[ -z "$PREFILL_SCHEDULE_INTERVAL" ] || require_positive_integer PREFILL_SCHEDULE_INTERVAL
case "$COMPILATION_LEVEL" in ""|0|1|2|3) : ;; *) die "COMPILATION_LEVEL must be empty or 0-3: $COMPILATION_LEVEL" ;; esac
case "$GENERATION_CONFIG" in auto|vllm) : ;; *) die "GENERATION_CONFIG must be auto or vllm: $GENERATION_CONFIG" ;; esac
[ -z "$FAIRNESS_ENGINE" ] || die "FAIRNESS_ENGINE=$FAIRNESS_ENGINE: --fairness-engine no longer exists on karmic-kraken; remove it and set PREFILL_COMPUTE_SHARE alone (with PREFILL_SCHEDULE_INTERVAL=1)"
case "$MAX_PARALLEL_PREFILLS" in ""|auto) : ;; *) require_positive_integer MAX_PARALLEL_PREFILLS ;; esac
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
if [ -n "$PREFILL_COMPUTE_SHARE" ]; then
  # KK SchedulerConfig: prefill_compute_share is a fraction in (0,1) or "auto",
  # and "cannot be combined with prefill_schedule_interval greater than one"
  # (vllm/config/scheduler.py). The half-life is auto-mode only.
  [ "$PREFILL_COMPUTE_SHARE" = auto ] \
    || awk -v v="$PREFILL_COMPUTE_SHARE" 'BEGIN{ exit !(v+0 > 0 && v+0 < 1 && v ~ /^[0-9]*\.?[0-9]+$/) }' \
    || die "PREFILL_COMPUTE_SHARE must be auto or a fraction in (0,1): $PREFILL_COMPUTE_SHARE"
  [ -z "$PREFILL_SCHEDULE_INTERVAL" ] || [ "$PREFILL_SCHEDULE_INTERVAL" = 1 ] \
    || die "PREFILL_COMPUTE_SHARE=$PREFILL_COMPUTE_SHARE requires PREFILL_SCHEDULE_INTERVAL=1 (the engine rejects share + interval > 1; the published GLM profile runs 0.4 / 1)"
  if [ -n "$PREFILL_COMPUTE_HALF_LIFE" ]; then
    [ "$PREFILL_COMPUTE_SHARE" = auto ] || die "PREFILL_COMPUTE_HALF_LIFE is only valid with PREFILL_COMPUTE_SHARE=auto"
    case "$PREFILL_COMPUTE_HALF_LIFE" in smooth|responsive) : ;; *) awk -v v="$PREFILL_COMPUTE_HALF_LIFE" 'BEGIN{ exit !(v+0 > 0 && v ~ /^[0-9]*\.?[0-9]+$/) }' || die "PREFILL_COMPUTE_HALF_LIFE must be smooth, responsive, or seconds > 0: $PREFILL_COMPUTE_HALF_LIFE" ;; esac
  fi
elif [ -n "$PREFILL_COMPUTE_HALF_LIFE" ]; then
  die "PREFILL_COMPUTE_HALF_LIFE requires PREFILL_COMPUTE_SHARE=auto"
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
  NCCL_IGNORE_CPU_AFFINITY NCCL_TUNER_PLUGIN CUTE_DSL_ARCH \
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

# MiMo checkpoint completeness (Luke's launcher checks these four): the b12x
# loader needs the index + tokenizer + config, and a missing shard surfaces
# only deep into the load otherwise.
for mf in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
  [ -f "$MODEL_HOST_PATH/$mf" ] \
    || die "MODEL_HOST_PATH has no $mf (complete the snapshot; the b12x loader needs it): $MODEL_HOST_PATH"
done

for name in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_CUDAGRAPH_CAPTURE_SIZE; do
  require_positive_integer "$name"
done
case "$NUM_SPECULATIVE_TOKENS" in
  ''|*[!0-9]*) die "NUM_SPECULATIVE_TOKENS must be a non-negative integer: $NUM_SPECULATIVE_TOKENS" ;;
esac
if [ "$SPECULATOR" = mtp ] && [ "$NUM_SPECULATIVE_TOKENS" -gt 3 ]; then
  warn "SPECULATOR=mtp with $NUM_SPECULATIVE_TOKENS tokens: MiMo ships 3 MTP layers and steps past 3 reuse them cyclically (mimo_v2_mtp.py: spec_step_idx % num_mtp_layers) -- expect tail acceptance to fall off"
fi
if [ "$SPECULATOR" = dflash ]; then
  [ -f "$MODEL_HOST_PATH/$DFLASH_SUBDIR/config.json" ] \
    || die "SPECULATOR=dflash but $MODEL_HOST_PATH/$DFLASH_SUBDIR/config.json is missing -- download the snapshot's dflash/ folder (hf download ... --include 'dflash/*')"
  [ "$NUM_SPECULATIVE_TOKENS" = 7 ] \
    || warn "SPECULATOR=dflash with $NUM_SPECULATIVE_TOKENS tokens: the draft's block_size is 8 (7 draft tokens); other depths are unqualified"
  case "$DFLASH_KV_CACHE_DTYPE" in auto|bfloat16|fp8) : ;; *) die "DFLASH_KV_CACHE_DTYPE must be auto, bfloat16, or fp8: $DFLASH_KV_CACHE_DTYPE" ;; esac
fi
if [ "$MAX_MODEL_LEN" = -1 ]; then
  # KK: -1 = auto-fit the context to the KV pool (ModelConfig.max_model_len ge=-1;
  # the published Spark preset runs -1 with an explicit KV_CACHE_MEMORY_BYTES).
  [ -n "$KV_CACHE_MEMORY_BYTES" ] \
    || die "MAX_MODEL_LEN=-1 (auto-fit) needs KV_CACHE_MEMORY_BYTES pinned; with profiling it auto-fits to whatever GPU_MEMORY_UTILIZATION leaves, which is not a reproducible contract"
  warn "MAX_MODEL_LEN=-1: the engine fits the context ceiling to the KV pin at startup; read the reported capacity from the boot log and pin an explicit value once qualified"
elif [ "$MAX_MODEL_LEN" != auto ]; then
  require_positive_integer MAX_MODEL_LEN
else
  warn "MAX_MODEL_LEN=auto resolves to the checkpoint maximum (1,048,576 for MiMo); on a Spark pair that profile can starve the KV pool -- the groundwork value is 262144"
fi

case "$LANGUAGE_MODEL_ONLY" in
  0|1) : ;;
  *) die "LANGUAGE_MODEL_ONLY must be 0 or 1: $LANGUAGE_MODEL_ONLY" ;;
esac
if [ "$LANGUAGE_MODEL_ONLY" = 0 ] && { [ "${LIMIT_MM:-1}" = 0 ] || [ "${MM_VIDEOS:-0}" -gt 0 ]; }; then
  warn "video is enabled (LIMIT_MM=0 or MM_VIDEOS>0): the encoder is profiled against a maximum-size VIDEO item (first boot: 114,688-token encoder budget, 6.88 GiB peak activation); MM_VIDEOS=0 with LIMIT_MM=1 drops it"
fi

require_port API_PORT
require_port MASTER_PORT
[ "$API_PORT" != "$MASTER_PORT" ] || die "API_PORT and MASTER_PORT must differ"

require_unit_fraction GPU_MEMORY_UTILIZATION

# Setting kv_cache_memory_bytes makes vLLM skip memory PROFILING (pool sizing
# comes from the pin), but GPU_MEMORY_UTILIZATION is NOT inert: the worker's
# request_memory() gate (vllm/v1/worker/utils.py) still refuses to start when
# CUDA-free-at-init < total x utilization -- on JJ and KK alike. Leave the pin
# empty for one boot to have vLLM profile at GPU_MEMORY_UTILIZATION and log a
# suggested value to pin. MiMo is a pure-attention hybrid (full + SWA
# layers; NO mamba/GDN recurrent state), so the pool is paged KV only -- but
# the SWA layers' checkpoints (prefix-cache retention) share the reservation.
if [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
  require_positive_integer KV_CACHE_MEMORY_BYTES
fi
# GB10 unified memory: a node reports ~121.7 GiB total with ~10.5 GiB held by
# the OS, so the gate refuses any utilization above roughly 0.913 on the cu132
# images. The NGC-based cu133 images (jovian r35-cu133, karmic-kraken) reserve
# ~13 GiB more at CUDA init: measured 2026-09-21, host MemFree 117.5 GiB but
# CUDA-free 108.5/121.69 at the gate, so 0.90 (109.5 GiB) failed by 1 GiB and
# 0.85 is the qualified value on that line.
awk -v v="$GPU_MEMORY_UTILIZATION" 'BEGIN{ exit !(v+0 > 0.90) }' \
  && warn "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION is at or past this node's free-memory ceiling (~0.913 on cu132 images); vLLM refuses at startup when the request exceeds CUDA-free memory"
case "$VLLM_NCCL_SO_PATH" in
  /opt/local-inference/nccl/*)
    awk -v v="$GPU_MEMORY_UTILIZATION" 'BEGIN{ exit !(v+0 > 0.88) }' \
      && warn "cu133-lineage image (NCCL under /opt/local-inference/nccl) with GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION: this line's CUDA init leaves ~108.5/121.69 GiB free (ceiling ~0.89); the request_memory gate failed at 0.90 on 2026-09-21. Qualified value: 0.85 (the KV pin still sizes the pool)" ;;
esac
{
  if command -v nvidia-smi >/dev/null 2>&1; then
    mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    case "$mem_total$mem_free" in
      ''|*[!0-9]*) ;;
      *)
        awk -v t="$mem_total" -v f="$mem_free" -v u="$GPU_MEMORY_UTILIZATION" \
          'BEGIN{ want = t*u; if (want > f) printf "mimo pair launcher: warning: GPU_MEMORY_UTILIZATION=%s asks for %.1f GiB but only %.1f GiB is free (ceiling %.4f); vLLM will refuse at startup\n", u, want/1024, f/1024, f/t > "/dev/stderr" }'
        ;;
    esac
  fi
}

for name in ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL TRUST_REMOTE_CODE \
  VLLM_ENABLE_PCIE_ALLREDUCE VLLM_B12X_MOE_FP4_FORCE_A16; do
  require_bool "$name"
done

case "$KV_CACHE_DTYPE" in
  bfloat16|auto) ;;
  fp8) warn "KV_CACHE_DTYPE=fp8 is UNVERIFIED for MiMo's 192-dim attention (the catalog qualifies bfloat16); treat as an A/B and check outputs" ;;
  *) die "KV_CACHE_DTYPE must be bfloat16, auto, or fp8: $KV_CACHE_DTYPE" ;;
esac
case "$CUDAGRAPH_MODE" in
  FULL|FULL_AND_PIECEWISE|PIECEWISE|NONE) ;;
  *) die "CUDAGRAPH_MODE must be FULL, FULL_AND_PIECEWISE, PIECEWISE, or NONE: $CUDAGRAPH_MODE" ;;
esac
case "$VLLM_KV_CACHE_LAYOUT" in
  ""|LBNHC|LBHNC|LHBNC|NHD|HND|BLHNC|BLNHC|BHLNC) : ;;
  *) die "VLLM_KV_CACHE_LAYOUT must be empty or a layout name (BLHNC for the mixed-page DFlash A/B): $VLLM_KV_CACHE_LAYOUT" ;;
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
[ "$NCCL_TUNER_PLUGIN" = none ] || die "NCCL_TUNER_PLUGIN must be none (no tuner plugin ships in the image; Luke's launcher and the published preset both pin none)"
# The image bakes NCCL_IB_DISABLE=1 for single-node PCIe boxes; the env file
# must override it back to 0 or the pair silently falls back to TCP sockets.
[ "$NCCL_IB_DISABLE" = 0 ] || die "NCCL_IB_DISABLE must be 0 (the serving image bakes 1 for single-node use; override it in the env file)"
# Fabric profiles:
#   single-path (validated here): MERGE_NICS=0, SUBNET_AWARE_ROUTING=0, one
#     HCA (rocep1s0f0 = enp1s0f0np0).
#   dual-path (Luke's Spark launchers, 2026-09-21): MERGE_NICS=1 with BOTH
#     Linux interfaces of the cabled QSFP port in NCCL_IB_HCA
#     (rocep1s0f0,roceP2p1s0f0). They are the two PCIe Gen5 x4 paths into
#     the same CX-7, not a second cable: one 200G link, striped over both
#     PCIe paths (Luke: 196 Gb/s combined vs a single-path cap). The second
#     interface needs its own IPv4 (own /24) so GID index 3 is populated;
#     RoCEnante picks up to two HCAs from NCCL_IB_HCA the same way.
#     SUBNET_AWARE_ROUTING is no longer part of that profile (dropped
#     upstream at 9e5d179); it only matters when the two paths sit on
#     different subnets behind a switch, so it is allowed either way.
if [ "$NCCL_IB_MERGE_NICS" = 1 ]; then
  case "$NCCL_IB_HCA" in
    *,*) : ;;
    *) warn "NCCL_IB_MERGE_NICS=1 with a single HCA in NCCL_IB_HCA -- dual-path wants both interfaces of the cabled port (e.g. rocep1s0f0,roceP2p1s0f0), each with its own IPv4 so GID $NCCL_IB_GID_INDEX exists on both" ;;
  esac
elif [ "$NCCL_IB_SUBNET_AWARE_ROUTING" != 0 ]; then
  warn "NCCL_IB_SUBNET_AWARE_ROUTING=$NCCL_IB_SUBNET_AWARE_ROUTING with NCCL_IB_MERGE_NICS=0: off the validated single-path profile (0/0); allowed but unmeasured here"
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
# The cu133 images ship NGC's forward-compat libcuda (R610, under
# /usr/local/cuda/compat/lib.real/). Preloading it on a DGX Spark takes every
# process down silently during torch's driver init (measured 2026-09-21,
# host driver 580.173.02): GB10 is not forward-compat hardware, and the 13.3
# toolkit runs on 580 under minor-version compatibility without it. The path
# exists in the image, so the image check below cannot catch this.
case ":$LD_PRELOAD:" in
  *"/cuda/compat/lib.real/"*) warn "LD_PRELOAD carries the NGC forward-compat libcuda (compat/lib.real); on this pair that crashed torch's CUDA init with no traceback (host driver 580.173.02 vs compat R610). cu133 images need only the NCCL preload: LD_PRELOAD=$VLLM_NCCL_SO_PATH" ;;
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
    stale="$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c '^mimo-v26-flash-r' || true)"
    if [ "${stale:-0}" -gt 0 ]; then
      die "only $(awk -v f="$mem_free_kib" 'BEGIN{printf "%.1f", f/1048576}') GiB free but utilization $GPU_MEMORY_UTILIZATION needs ~$(awk -v r="$required_kib" 'BEGIN{printf "%.1f", r/1048576}') GiB (short $deficit_gib GiB) -- a mimo-v26-flash container is already running on this host; --down it first"
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
model_container_path=/models/mimo-v2.6-flash-rl

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
extra_args=()
[ -z "$PREFILL_SCHEDULE_INTERVAL" ] || extra_args+=(--prefill-schedule-interval "$PREFILL_SCHEDULE_INTERVAL")
case "$INSTANTTENSOR_COPY" in
  auto) : ;;
  0|1) warn "INSTANTTENSOR_COPY=$INSTANTTENSOR_COPY passes instanttensor_copy to the loader; vllm pins >= 6575b5ac (2026-09-05) removed that option (launchers moved to --load-format fastsafetensors)" ;;&
  0) extra_args+=(--model-loader-extra-config '{"instanttensor_copy":false}') ;;
  1) extra_args+=(--model-loader-extra-config '{"instanttensor_copy":true}') ;;
  *) die "INSTANTTENSOR_COPY must be auto, 0, or 1" ;;
esac
[ -z "$COMPILATION_LEVEL" ] || extra_args+=("-O$COMPILATION_LEVEL")
if [ "$ASYNC_SCHEDULING" = 1 ]; then extra_args+=(--async-scheduling); fi
[ -z "$PREFIX_RETENTION_INTERVAL" ] || extra_args+=(--prefix-cache-retention-interval "$PREFIX_RETENTION_INTERVAL")
[ -z "$PREFILL_COMPUTE_SHARE" ] || extra_args+=(--prefill-compute-share "$PREFILL_COMPUTE_SHARE")
[ -z "$MM_PROCESSOR_CACHE_GB" ] || extra_args+=(--mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB")
[ -z "$MM_ENCODER_TP_MODE" ] || extra_args+=(--mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")
# generation-config: default vllm is emitted literally in the command;
# only emit here when set to something else.
chat_template_container_path=/models/chat_template.jinja
[ -z "$CHAT_TEMPLATE_HOST_PATH" ] || extra_args+=(--chat-template "$chat_template_container_path")

quantization_args=()
if [ "$QUANTIZATION" != auto ]; then
  quantization_args=(--quantization "$QUANTIZATION")
fi

# Sampling overrides: empty = Luke's launcher literal (1.0/0.95); any knob set
# builds the JSON instead (the literal flag is emitted as ${sampling_override_args}).
sampling_json=""
[ -z "$SAMPLING_TEMPERATURE" ] || sampling_json="\"temperature\":$SAMPLING_TEMPERATURE"
[ -z "$SAMPLING_TOP_P" ] || sampling_json="${sampling_json:+$sampling_json,}\"top_p\":$SAMPLING_TOP_P"
[ -z "$SAMPLING_TOP_K" ] || sampling_json="${sampling_json:+$sampling_json,}\"top_k\":$SAMPLING_TOP_K"
[ -z "$SAMPLING_MIN_P" ] || sampling_json="${sampling_json:+$sampling_json,}\"min_p\":$SAMPLING_MIN_P"
[ -z "$SAMPLING_REPETITION_PENALTY" ] || sampling_json="${sampling_json:+$sampling_json,}\"repetition_penalty\":$SAMPLING_REPETITION_PENALTY"
if [ -n "$sampling_json" ]; then
  sampling_override_args=(--override-generation-config "{$sampling_json}")
else
  sampling_override_args=(--override-generation-config '{"temperature":1.0,"top_p":0.95}')
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
      # Luke's MiMo launcher shape: the in-checkpoint MTP head needs ONLY
      # method + depth (no backend keys, no sampling keys -- the engine
      # default greedy applies; the catalog leaves speculators unqualified).
      speculative_config=$(printf '{"method":"mtp","num_speculative_tokens":%s}' "$NUM_SPECULATIVE_TOKENS")
      ;;
    dflash)
      # KK serve-mimo26-pro.sh shape: separate draft snapshot, draft KV auto,
      # B12X attention. The draft lives inside the read-only model mount.
      speculative_config=$(printf '{"method":"dflash","model":"%s/%s","num_speculative_tokens":%s,"kv_cache_dtype":"%s","attention_backend":"B12X"}' \
        "$model_container_path" "$DFLASH_SUBDIR" "$NUM_SPECULATIVE_TOKENS" "$DFLASH_KV_CACHE_DTYPE")
      ;;
  esac
  speculative_args=(--speculative-config "$speculative_config")
fi

# Catalog compilation contract for MiMo: custom_ops none (GLM/Qwen force
# ["all"]; MiMo's qualified config does not).
# KK's serve-mimo26-flash.sh passes NO compilation config: custom_ops keeps
# the engine default. Forcing ["none"] would turn off vLLM's custom CUDA ops
# (rms_norm etc. fall back to native torch) on a CUDA-graph-only run, so it
# is emitted only when CUSTOM_OPS is set explicitly.
if [ -n "${CUSTOM_OPS:-}" ]; then
  compilation_config=$(printf '{"cudagraph_mode":"%s","custom_ops":["%s"]}' "$CUDAGRAPH_MODE" "$CUSTOM_OPS")
else
  compilation_config=$(printf '{"cudagraph_mode":"%s"}' "$CUDAGRAPH_MODE")
fi

case "$CONTAINER_RUNTIME" in
  podman) gpu_args=(--device nvidia.com/gpu=all --security-opt label=disable) ;;
  *)      gpu_args=(--gpus all) ;;
esac

mount_args=(
  -v "$MODEL_HOST_PATH:$model_container_path:ro"
  -v "$CACHE_HOST_PATH:/cache"
)
if [ -n "$CHAT_TEMPLATE_HOST_PATH" ]; then
  mount_args+=(-v "$CHAT_TEMPLATE_HOST_PATH:$chat_template_container_path:ro")
fi
if [ -n "$TORCH_PROFILE_HOST_DIR" ]; then
  mkdir -p "$TORCH_PROFILE_HOST_DIR"
  mount_args+=(-v "$TORCH_PROFILE_HOST_DIR:/profiles")
fi

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
  ${CONTAINER_MEMORY_GB:+--memory ${CONTAINER_MEMORY_GB}g --memory-swap $((${CONTAINER_MEMORY_GB:-0}+4))g}
  --device /dev/infiniband
  "${seccomp_args[@]}"   # b12x loader io_uring (shared block above)
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

  # --- model / kernels (catalog contract: b12x everything, BF16 KV) --------
  --dtype bfloat16
  "${quantization_args[@]}"
  --attention-backend "$ATTENTION_BACKEND"
  --moe-backend "$MOE_BACKEND"
  --linear-backend "$LINEAR_BACKEND"
  "${lm_only_args[@]}"
  --load-format "$LOAD_FORMAT"
  --compilation-config "$compilation_config"
  --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"

  # --- scheduling ---------------------------------------------------------
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  "${extra_args[@]}"

  # --- model behaviour (catalog: mimo parsers, generation-config vllm) -----
  --reasoning-parser mimo
  --tool-call-parser mimo
  --enable-auto-tool-choice
  --generation-config "$GENERATION_CONFIG"
  "${sampling_override_args[@]}"
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
printf '  cache:                   %s\n' "$CACHE_HOST_PATH"
printf '  MAX_MODEL_LEN:           %s\n' "$MAX_MODEL_LEN"
printf '  MAX_NUM_SEQS:            %s\n' "$MAX_NUM_SEQS"
printf '  MAX_NUM_BATCHED_TOKENS:  %s\n' "$MAX_NUM_BATCHED_TOKENS"
printf '  SPECULATOR:              %s (%s draft tokens)\n' \
  "$SPECULATOR" "$NUM_SPECULATIVE_TOKENS"
if [ -n "$PREFILL_COMPUTE_SHARE" ]; then
  printf '  PREFILL:                 compute share %s, interval %s%s\n' "$PREFILL_COMPUTE_SHARE" "$PREFILL_SCHEDULE_INTERVAL" "${MAX_PARALLEL_PREFILLS:+, max parallel $MAX_PARALLEL_PREFILLS}"
else
  printf '  PREFILL:                 schedule interval %s%s\n' "${PREFILL_SCHEDULE_INTERVAL:-engine default}" "${MAX_PARALLEL_PREFILLS:+, max parallel $MAX_PARALLEL_PREFILLS}"
fi
printf '  KV_CACHE_MEMORY_BYTES:   %s (%s)\n' \
  "${KV_CACHE_MEMORY_BYTES:-profiled at $GPU_MEMORY_UTILIZATION}" "$KV_CACHE_DTYPE"
if [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
  printf '  GPU_MEMORY_UTILIZATION:  %s (startup free-memory gate only; pool size comes from the pin)\n' "$GPU_MEMORY_UTILIZATION"
else
  printf '  GPU_MEMORY_UTILIZATION:  %s\n' "$GPU_MEMORY_UTILIZATION"
fi
printf '  CUDAGRAPH_MODE:          %s (capture %s)\n' \
  "$CUDAGRAPH_MODE" "$MAX_CUDAGRAPH_CAPTURE_SIZE"
printf '  LOAD_FORMAT:             %s\n' "$LOAD_FORMAT"
printf '  seccomp:                 %s\n' "$seccomp_desc"
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
