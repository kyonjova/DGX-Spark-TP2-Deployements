#!/usr/bin/env bash
# template_pair_serve.sh -- model-agnostic launcher for ONE RANK of a two-node
# DGX Spark pair (TP=2 across a direct ConnectX-7 RoCE link).
#
# TARGET STACK, not generic vLLM. This launcher assumes an aarch64/sm_121
# image built by build-spark-cu132.sh from:
#   - local-inference-lab/vllm, branch dev/jovian-judgement
#   - local-inference-lab/sparkinfer (B12X) kernels
#   - the patched NCCL + CUDA compat shim that image carries
# Many flags emitted below (--attention-backend B12X, --moe-backend b12x,
# --linear-backend, --kda-prefill-backend, --gdn-decode-kernel,
# --recurrent-checkpoint-policy, --fairness-engine, --prefill-schedule-interval)
# exist only on that branch and only at certain pins. Every one of them is
# omitted when its variable is empty, so this file still runs against an older
# image -- but stock upstream vLLM will reject the B12X backends outright.
#
# Distilled from glm53_pair_serve.sh (newest; authoritative where the two
# differ) and deepseek_v4_pair_serve.sh. Everything that named a model was
# moved into the MODEL BLOCK below as data or a hook function. The body after
# the block is shared pair plumbing and should not need edits for a new model.
#
# To instantiate for a new model:
#   1. cp template_pair_serve.sh MODEL_pair_serve.sh
#   2. fill the MODEL BLOCK -- it is the ONLY section that changes. The script
#      refuses to run while any angle-bracket placeholder remains in it.
#   3. cp template-rank.env.example MODEL-rank.env.example; fill section 7 with
#      the same keys model_validate() checks.
#   4. iterate ./MODEL_pair_serve.sh --check rank-0.env until it prints the
#      container command; then --run rank 1 first, rank 0 second.
#
# Design contract (unchanged from the source launchers):
#   - the per-rank env file is BOTH the host launch contract (sourced here) and
#     the container environment (--env-file). One home per value. This is why
#     values are unquoted and lists are comma- not space-separated: shell
#     sourcing strips quotes, docker --env-file does not.
#   - everything is validated at --check time. Nothing fails minutes into a
#     boot that a stat() or a grep could have caught on the host.
#   - rank 1 (--headless) starts first and blocks on rank 0's rendezvous.
#
# Usage:
#     ./MODEL_pair_serve.sh --check   rank-0.env   # validate + print command (default)
#     ./MODEL_pair_serve.sh --run     rank-0.env   # validate, then start detached
#     ./MODEL_pair_serve.sh --restart rank-0.env   # --down, then --run
#     ./MODEL_pair_serve.sh --logs    rank-0.env   # follow this rank's container
#     ./MODEL_pair_serve.sh --status  rank-0.env   # this rank's container state
#     ./MODEL_pair_serve.sh --verify  rank-0.env   # grep the log for startup markers
#     ./MODEL_pair_serve.sh --down    rank-0.env   # stop + rm this rank's container
#     ./MODEL_pair_serve.sh --clear   rank-0.env   # wipe CACHE_HOST_PATH (guarded)
# Anything after ENV_FILE is appended verbatim to the vllm serve argv.

set -euo pipefail

# MODEL-BLOCK-BEGIN =============================================================
# MODEL BLOCK -- the only section a new model edits.
#
# Data first, then hooks. Each hook is called exactly once by the body at the
# point tagged "hook:" further down. Hooks may rely on the env file already
# being sourced and on the common validators (require_*) being defined.
# ==============================================================================

# Short slug used as the log prefix and the container name (SLUG-rRANK).
MODEL_ID='<model_slug>'

# Read-only mount point of MODEL_HOST_PATH inside the container.
MODEL_CONTAINER_PATH='/models/<model_dir>'

# Read-only mount point of DRAFT_MODEL_HOST_PATH; used only when
# model_needs_draft returns true for the selected SPECULATOR.
DRAFT_CONTAINER_PATH='/models/<draft_dir>'

# Legal SPECULATOR values, space-separated. "none" must be present.
# Examples: "mtp dflash2 none" (GLM-5.3), "dspark none" (DeepSeek-V4).
SPECULATOR_CHOICES="none"

# Legal KV_CACHE_DTYPE values for this model's attention backend. Newer pins
# add packed layouts: fp8_ds_mla (r26) and nvfp4_ds_mla (half the bytes,
# quality-gated). Older images reject them.
# Examples: "fp8 auto", "fp8 fp8_ds_mla nvfp4_ds_mla auto".
KV_CACHE_DTYPE_CHOICES="fp8 auto"

# 1 if the checkpoint is a vision-language variant. Gates the whole multimodal
# block (--language-model-only, --limit-mm-per-prompt, --mm-processor-cache-gb,
# --mm-encoder-tp-mode); at 0 those flags are never emitted and the launcher
# warns if the env file sets them.
MODEL_MULTIMODAL=0

# Cosmetic: printed while hashing a draft checkpoint so the pause is explained.
DRAFT_SIZE_HINT="~1 GB"

# --verify marker regex: the startup log lines that prove the QUALIFIED path
# was taken, not merely that the server came up. Absence of a marker you
# expect (a kernel family, a speculator line, a page-split line) means a
# fallback path ran. Keep the generic engine lines and add the model's own.
VERIFY_LOG_MARKERS='<extend_with_model_specific_markers>|speculative_config|Using .* all-reduce backends|attention block size|Available KV cache memory|GPU KV cache size|Graph capturing finished|cudagraph_mode=|Application startup complete'

# hook: per-model fallbacks. Runs AFTER the env file is sourced, so `: ${X:=}`
# fills only what the file left empty. Every model default lives here; the
# body carries no model values.
model_defaults() {
  : "${SERVED_MODEL_NAME:=<served_model_name>}" # comma-separated aliases; first is what /v1/models reports
  : "${SPECULATOR:=none}"                        # one of SPECULATOR_CHOICES
  : "${ATTENTION_BACKEND:=B12X}"                 # --attention-backend
  : "${MOE_BACKEND:=b12x}"                       # --moe-backend
  : "${KV_CACHE_DTYPE:=fp8}"                     # one of KV_CACHE_DTYPE_CHOICES
  : "${GPU_MEMORY_UTILIZATION:=0.88}"            # profiling fraction; ignored once KV_CACHE_MEMORY_BYTES is pinned
  : "${MAX_MODEL_LEN:=<measured_ceiling>}"       # measure on the pair (env template, section 2)
  : "${MAX_NUM_SEQS:=8}"                         # concurrency ceiling; see decode_batch check
  : "${MAX_NUM_BATCHED_TOKENS:=4096}"            # engine step / prefill chunk; measure per model (GLM-5.3: 4096 flat ~1.9k prefill; 8192 peaks higher but cliffs at 128k)
  : "${MAX_CUDAGRAPH_CAPTURE_SIZE:=96}"          # largest captured decode batch
  : "${CUDAGRAPH_MODE:=FULL_AND_PIECEWISE}"      # FULL demotes to FULL_DECODE_ONLY on hybrid (GDN) backends
  : "${LOAD_FORMAT:=instanttensor}"              # --load-format; instanttensor ~2.4 GB/s on Spark NVMe
  : "${TRUST_REMOTE_CODE:=0}"                    # 1 only if the checkpoint needs remote modeling code
  # Speculative depth default depends on the speculator chosen above.
  case "$SPECULATOR" in
    none) : "${NUM_SPECULATIVE_TOKENS:=0}" ;;
    *)    : "${NUM_SPECULATIVE_TOKENS:=<default_depth>}" ;;
  esac
}

# hook: does the selected SPECULATOR load a separate draft checkpoint?
# return 0 = yes (DRAFT_MODEL_HOST_PATH required, mounted, optionally hashed).
model_needs_draft() {
  case "$SPECULATOR" in
    # dflash2) return 0 ;;   # GLM-5.3: separate DFlash2 draft
    # dspark)  return 0 ;;   # DeepSeek-V4: draft head lives in the main checkpoint -> return 1
    *) return 1 ;;
  esac
}

# hook: model-specific validation. Runs after the common validators, so every
# common integer/bool is already checked and usable in arithmetic. Put
# required keys, kernel-toggle sanity, and KNOWN-BAD combinations here; each
# one should cite the boot that proved it.
model_validate() {
  : # TODO(model): replace with real checks. Examples, from the source launchers:
  # for name in VLLM_B12X_MOE_FP4_FORCE_A16; do require_value "$name"; require_bool "$name"; done
  # [ "$VLLM_B12X_MOE_FP4_FORCE_A16" = 0 ] || warn "A16 is the pre-2026-08-28 W4A16 path; qualified recipe runs 0"
  # case "$LINEAR_BACKEND" in b12x) : ;; *) warn "LINEAR_BACKEND=$LINEAR_BACKEND is off the qualified recipe" ;; esac
  # [ "$SPECULATOR" != dflash2 ] || [ "$NUM_SPECULATIVE_TOKENS" = 7 ] || warn "DFlash2 is trained for 7 (8-token block)"
  # [ "$SPECULATOR" != dspark ] || [ "$NUM_SPECULATIVE_TOKENS" -ge 5 ] || die "DSpark below one 5-token block is INCORRECT OUTPUT"
  # [ "$B12X_MOE_FORCE_A8" != 1 ] || [ "${VLLM_MOE_SKIP_PADDING:-0}" != 1 ] || die "A8 + SKIP_PADDING faulted cudagraph capture (2026-08-26)"
}

# helper for hooks: the adaptive-depth JSON fields (leading comma) or nothing.
# Append the result inside the speculative-config object for mtp-family
# speculators that support adaptive depth (GLM-5.3 mtp does; DFlash does not).
adaptive_speculative_fields() {
  [ "$ADAPTIVE_SPECULATIVE_TOKENS" = 1 ] || return 0
  printf ',"adaptive_speculative_tokens_window":%s,"adaptive_speculative_tokens_initial":%s' \
    "$ADAPTIVE_SPECULATIVE_TOKENS_WINDOW" "$ADAPTIVE_SPECULATIVE_TOKENS_INITIAL"
}

# hook: print the --speculative-config JSON for the selected SPECULATOR.
# Called only when SPECULATOR != none and NUM_SPECULATIVE_TOKENS > 0.
# Build JSON with printf HERE, never in the env file (sourcing strips quotes).
model_speculative_config() {
  case "$SPECULATOR" in
    # Read the draft's kernel backends from the env so they are A/B-able
    # without editing the launcher:
    # mtp)     printf '{"method":"mtp","num_speculative_tokens":%s,"moe_backend":"%s","attention_backend":"%s"%s}' "$NUM_SPECULATIVE_TOKENS" "${MTP_MOE_BACKEND:-humming}" "${MTP_ATTENTION_BACKEND:-B12X}" "$(adaptive_speculative_fields)" ;;
    # dflash2) printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"kv_cache_dtype":"auto"}' "$DRAFT_CONTAINER_PATH" "$NUM_SPECULATIVE_TOKENS" ;;
    # dspark)  printf '{"model":"%s","method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"greedy"}' "$MODEL_CONTAINER_PATH" "$NUM_SPECULATIVE_TOKENS" ;;
    *) die "model_speculative_config: no JSON defined for SPECULATOR=$SPECULATOR" ;;
  esac
}

# hook: append model-specific vllm serve flags to model_args. The body already
# emits topology, memory, cudagraph/compilation, scheduling, speculative,
# served-name and observability flags; put only what is true of THIS model.
model_serve_args() {
  model_args+=(
    # --dtype bfloat16                    # activation dtype for the non-quantized path
    # --quantization modelopt_mixed       # omit to let the checkpoint self-describe
    # --linear-backend b12x               # pin quantized linears to sparkinfer kernels
    # --mamba-cache-mode align            # hybrid GDN models: aligned state pages in the KV reservation
    # --language-model-only               # VL checkpoints: skip encoder-cache reservation during profiling
    # (FlashInfer autotune is the ENABLE_FLASHINFER_AUTOTUNE knob, not a recipe line)
    # --tokenizer-mode deepseek_v4        # model-family tokenizer
    # --reasoning-parser glm45            # model-family reasoning parser
    # --tool-call-parser glm47            # model-family tool-call parser
    # --enable-auto-tool-choice
    # --async-scheduling --no-scheduler-reserve-full-isl   # DeepSeek recipe
  )
}

# hook: append extra `-e KEY=VALUE` container env to container_env, for values
# that must beat both the env file and the image's baked ENV.
model_container_env() {
  : # e.g. container_env+=(-e SOME_BAKED_VAR=override)
}

# hook: print the --default-chat-template-kwargs JSON body WITHOUT the braces,
# or nothing. The kwarg names come from the checkpoint's own chat template, so
# they cannot live in the common body. Build with printf -- JSON in the env
# file loses its quotes when the file is sourced. Per-request values still win.
model_chat_template_kwargs() {
  : # e.g. GLM-5.3: reasoning_effort + clear_thinking
  # local out=""
  # [ -z "${REASONING_EFFORT-}" ] || out="\"reasoning_effort\":\"$REASONING_EFFORT\""
  # [ -z "${CLEAR_THINKING-}" ] || { ct=false; [ "$CLEAR_THINKING" = 1 ] && ct=true
  #   out="${out:+$out,}\"clear_thinking\":$ct"; }
  # printf '%s' "$out"
  # e.g. DeepSeek-V4: "thinking":true,"reasoning_effort":"high"
}

# hook: extra lines for the --check summary (printf, 2-space indent, 25-col key).
model_summary() {
  : # e.g. printf '  MoE activation mode:     %s\n' "$mode"
}
# MODEL-BLOCK-END ===============================================================


# ------------------------------------------------------------------ plumbing

die() {
  echo "${MODEL_ID} pair launcher: $*" >&2
  exit 20
}

warn() {
  echo "${MODEL_ID} pair launcher: warning: $*" >&2
}

# Refuse to run as the template. Strip comments from the MODEL BLOCK and look
# for an unresolved angle-bracket placeholder in code or data.
self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
if sed -n '/^# MODEL-BLOCK-BEGIN/,/^# MODEL-BLOCK-END/p' "$self" \
   | sed 's/[[:space:]]*#.*$//' | grep -Eq '<[A-Za-z0-9_]+>'; then
  echo "pair launcher: MODEL BLOCK still contains placeholders; this is the template, not a model launcher" >&2
  exit 20
fi
case " $SPECULATOR_CHOICES " in
  *" none "*) ;;
  *) die "SPECULATOR_CHOICES must include none: $SPECULATOR_CHOICES" ;;
esac

usage() {
  cat >&2 <<EOF
usage: $(basename "$0") [MODE] ENV_FILE [extra vllm args...]

  --check     validate the env file and print the launch command (default)
  --run       validate, then start the container detached
  --restart   stop and remove this rank's container, then run
  --down      stop and remove this rank's container
  --logs      follow this rank's container logs
  --status    show this rank's container state
  --verify    grep this rank's log for the startup markers (+ /health on rank 0)
  --clear     wipe CACHE_HOST_PATH contents (container must not exist)
EOF
}

# ---------------------------------------------------------------- arguments

mode=--check
case "${1:-}" in
  --check|--run|--restart|--down|--logs|--status|--clear) mode=$1; shift ;;
  -h|--help)     usage; exit 0 ;;
  --*)           usage; exit 64 ;;
esac

env_file=${1:-}
[ -n "$env_file" ] || { usage; exit 64; }
shift
passthrough=("$@")

[ -f "$env_file" ] || die "environment file is missing: $env_file"
# Canonicalise now: later code cd's and docker --env-file needs the real path.
env_file=$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")

# CRLF breaks the shell source and docker --env-file silently and differently.
if grep -qU $'\r' "$env_file" 2>/dev/null; then
  die "environment file has CRLF line endings; convert with: sed -i 's/\r$//' $env_file"
fi

# Unresolved placeholders in non-comment lines are refused before sourcing.
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

: "${CONTAINER_RUNTIME:=docker}"        # docker | podman
: "${CONTAINER_NAME_SUFFIX:=}"          # lets two env files coexist on one host (A/B)
mgmt_container="${MODEL_ID}-r${NODE_RANK:-?}${CONTAINER_NAME_SUFFIX}"

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

  # rm -rf against a bad value is unrecoverable, so the target must look like
  # a cache directory: absolute, existing, writable, not a system root, at
  # least four path levels deep, and not one of the model directories.
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
  [ "$target" != "${DRAFT_MODEL_HOST_PATH-}" ] \
    || die "CACHE_HOST_PATH equals DRAFT_MODEL_HOST_PATH; refusing to clear: $target"

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
  # `find | head` races under pipefail (head exits, find takes SIGPIPE); contain it.
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
  --clear)  container_clear; exit 0 ;;
  --down)   container_down;  exit 0 ;;
  --logs)   require_runtime; exec "$CONTAINER_RUNTIME" logs -f "$mgmt_container" ;;
  --verify)
    # Startup markers, not a health check: a server that answers /health can
    # still have fallen back to a slow kernel path. Run on BOTH ranks -- the
    # worker logs its own kernel selection and page-split lines.
    require_runtime
    "$CONTAINER_RUNTIME" logs "$mgmt_container" 2>&1 \
      | grep -E "$VERIFY_LOG_MARKERS" | sed 's/^/  /'
    if [ "${NODE_RANK:-}" = 0 ] && command -v curl >/dev/null 2>&1; then
      printf 'health: '
      curl -fsS "http://127.0.0.1:${API_PORT:-8000}/health" >/dev/null 2>&1 \
        && echo OK || echo "not ready"
    fi
    exit 0
    ;;
  --status)
    require_runtime
    "$CONTAINER_RUNTIME" ps -a --filter "name=^/${mgmt_container}$" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'
    exit 0
    ;;
  --restart) container_down; mode=--run ;;
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

# ------------------------------------------------------- common defaults
# Pair-generic fallbacks. Model defaults come from the hook, which runs first
# so the model can override any of these.

model_defaults   # hook: model fallbacks

: "${KV_CACHE_MEMORY_BYTES:=}"               # empty = vLLM profiles at GPU_MEMORY_UTILIZATION
: "${ENABLE_PREFIX_CACHING:=1}"
: "${ENABLE_CHUNKED_PREFILL:=1}"
: "${ENABLE_FLASHINFER_AUTOTUNE:=}"          # empty = omit both flags (engine default); 0 = --no-enable-flashinfer-autotune; 1 = --enable-flashinfer-autotune
# Adaptive speculative depth (vLLM mtp-family speculators; pin >= e10536aa):
# the engine varies depth between 1 and NUM_SPECULATIVE_TOKENS from measured
# acceptance. The hook appends the fields via adaptive_speculative_fields.
: "${ADAPTIVE_SPECULATIVE_TOKENS:=0}"        # 0 = fixed depth (GLM-5.3: static 3 measured better at c1-c4); 1 = adaptive
: "${ADAPTIVE_SPECULATIVE_TOKENS_INITIAL:=}" # starting depth when adaptive; empty = min(3, NUM_SPECULATIVE_TOKENS)
: "${ADAPTIVE_SPECULATIVE_TOKENS_WINDOW:=32}" # verification steps per depth adjustment
: "${SERVING_IMAGE:=}"                       # required: image tag or digest present on BOTH nodes
: "${MEM_PREFLIGHT:=die}"                    # die | warn | off -- host free-memory gate below
: "${FABRIC_PROFILE:=single}"                # single | dual -- see the fabric checks below
# Scheduler/loader/profiling knobs. All default to "omit the flag" so the
# template runs on an image that has never heard of them; an unknown CLI flag
# is an argparse error at exec, unlike an unknown env var which only warns.
: "${BLOCK_SIZE:=256}"                        # paged-KV block for the public block table; hybrids override the physical page anyway
: "${API_KEY:=}"                             # empty = open port; set to require Authorization: Bearer on rank 0
: "${PREFILL_SCHEDULE_INTERVAL:=}"           # N = --prefill-schedule-interval N (needs vLLM PR #546); stops chunked prefill starving decode
: "${FAIRNESS_ENGINE:=}"                     # compute_share | micro_slicing -- r26 scheduler, an ALTERNATIVE to the interval above
: "${PREFILL_COMPUTE_SHARE:=}"               # fraction of contended execution given to prefill; requires FAIRNESS_ENGINE=compute_share
: "${RECURRENT_CHECKPOINT_POLICY:=}"         # hybrid/recurrent models only: auto | request_boundaries | aligned (prefix-cache state retention)
: "${KDA_PREFILL_BACKEND:=}"                 # GDN/KDA models only: auto | triton | flashkda | b12x
: "${GDN_DECODE_KERNEL:=}"                   # GDN/KDA models only: b12x | cuda | triton
# Sampling defaults pushed into the engine via --override-generation-config.
# Empty = whatever GENERATION_CONFIG resolves to. Per-request fields still win.
: "${SAMPLING_TEMPERATURE:=}"
: "${SAMPLING_TOP_P:=}"
: "${SAMPLING_TOP_K:=}"
: "${SAMPLING_MIN_P:=}"
: "${SAMPLING_REPETITION_PENALTY:=}"
# Multimodal, honoured only when MODEL_MULTIMODAL=1.
: "${LANGUAGE_MODEL_ONLY:=1}"                # 1 skips the vision encoder cache + max-size video profile item
: "${LIMIT_MM:=1}"                           # 1 = emit --limit-mm-per-prompt from the caps below; 0 = engine default (999 per modality)
: "${MM_IMAGES:=4}"                          # per-PROMPT image cap (the whole chat history counts), 0 disables the modality
: "${MM_VIDEOS:=0}"                          # per-PROMPT video cap; 0 drops the max-size video profile item
: "${MM_PROCESSOR_CACHE_GB:=}"               # host-RAM cache of preprocessed media; on unified memory this is KV memory. 0 disables
: "${MM_ENCODER_TP_MODE:=}"                  # data = each rank encodes its own items (no encoder collectives) | weights = engine default
: "${COMPILATION_LEVEL:=}"                   # 0-3 = -O<N> torch.compile level; empty = engine default. A/B only
: "${GENERATION_CONFIG:=auto}"               # auto = checkpoint's generation_config.json; vllm = engine defaults
: "${CHAT_TEMPLATE_HOST_PATH:=}"             # optional .jinja mounted read-only and passed as --chat-template
: "${INSTANTTENSOR_COPY:=auto}"              # 0 = skip loader staging copies (bounded unified-memory load); needs a pin that knows the key
: "${TORCH_PROFILE_HOST_DIR:=}"              # host dir -> /profiles + VLLM_TORCH_PROFILER_DIR, enabling /start_profile + /stop_profile
: "${CONTAINER_MEMORY_GB:=}"                 # cgroup cap so a runaway load cannot take the host down
: "${SHM_SIZE:=16g}"                         # inert under --ipc host; kept for runtimes that honour it
# InstantTensor TP=2 staging bounds (caps the checkpoint-loading peak).
: "${INSTANTTENSOR_BUFFER_SIZE:=1342177280}" # must exceed the checkpoint's largest tensor or the loader enlarges it every boot
: "${INSTANTTENSOR_IO_DEPTH:=3}"
: "${INSTANTTENSOR_CONCURRENCY:=1}"
: "${INSTANTTENSOR_CHUNK_SIZE:=8388608}"

# ------------------------------------------------------------ launch checks

for name in \
  NODE_RANK MASTER_ADDR MODEL_HOST_PATH CACHE_HOST_PATH SERVING_IMAGE \
  API_PORT MASTER_PORT MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
  ATTENTION_BACKEND MOE_BACKEND LOAD_FORMAT \
  LD_PRELOAD VLLM_NCCL_SO_PATH NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME \
  VLLM_HOST_IP NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_IB_HCA \
  NCCL_IB_GID_INDEX NCCL_IB_SUBNET_AWARE_ROUTING NCCL_IB_MERGE_NICS \
  NCCL_PROTO NCCL_P2P_LEVEL NCCL_CROSS_NIC NCCL_CUMEM_ENABLE \
  NCCL_IGNORE_CPU_AFFINITY CUTE_DSL_ARCH VLLM_ENABLE_PCIE_ALLREDUCE; do
  require_value "$name"
done

case "$NODE_RANK" in
  0|1) ;;
  *) die "NODE_RANK must be 0 or 1: $NODE_RANK" ;;
esac

case " $SPECULATOR_CHOICES " in
  *" $SPECULATOR "*) ;;
  *) die "SPECULATOR must be one of: $SPECULATOR_CHOICES (got $SPECULATOR)" ;;
esac

require_directory MODEL_HOST_PATH
require_directory CACHE_HOST_PATH
[ -r "$MODEL_HOST_PATH" ] || die "MODEL_HOST_PATH is not readable: $MODEL_HOST_PATH"
# A wrong-but-existing mount fails obscurely in the container and HF_HUB_OFFLINE=1
# removes any chance of recovery, so check for actual model bytes.
[ -f "$MODEL_HOST_PATH/config.json" ] \
  || die "MODEL_HOST_PATH has no config.json; wrong directory or a broken mount: $MODEL_HOST_PATH"
ls "$MODEL_HOST_PATH"/*.safetensors >/dev/null 2>&1 \
  || die "MODEL_HOST_PATH contains no weight files (*.safetensors): $MODEL_HOST_PATH"
[ -w "$CACHE_HOST_PATH" ] || die "CACHE_HOST_PATH is not writable: $CACHE_HOST_PATH"

# Separate draft checkpoint: existence, bytes, and (optionally) content hash.
# Draft weights drift upstream without a name change; only a hash proves the
# two ranks hold the same draft. Unset = warn, set = enforce.
if model_needs_draft; then   # hook
  require_value DRAFT_MODEL_HOST_PATH
  require_directory DRAFT_MODEL_HOST_PATH
  [ -f "$DRAFT_MODEL_HOST_PATH/config.json" ] \
    || die "DRAFT_MODEL_HOST_PATH has no config.json: $DRAFT_MODEL_HOST_PATH"
  ls "$DRAFT_MODEL_HOST_PATH"/*.safetensors >/dev/null 2>&1 \
    || die "DRAFT_MODEL_HOST_PATH contains no weight files: $DRAFT_MODEL_HOST_PATH"
  if [ -n "${DRAFT_WEIGHTS_SHA256-}" ]; then
    printf '%s' "$DRAFT_WEIGHTS_SHA256" | grep -Eq '^[0-9a-f]{64}$' \
      || die "DRAFT_WEIGHTS_SHA256 must be 64 lowercase hex chars: $DRAFT_WEIGHTS_SHA256"
    draft_weight_count=$(ls "$DRAFT_MODEL_HOST_PATH"/*.safetensors | wc -l)
    [ "$draft_weight_count" -eq 1 ] \
      || die "DRAFT_WEIGHTS_SHA256 pins a single-file draft; found $draft_weight_count *.safetensors in $DRAFT_MODEL_HOST_PATH"
    draft_weights=$(ls "$DRAFT_MODEL_HOST_PATH"/*.safetensors)
    echo "${MODEL_ID} pair launcher: hashing draft weights (${DRAFT_SIZE_HINT}, a few seconds): $draft_weights" >&2
    draft_weights_actual=$(sha256sum "$draft_weights" | cut -d' ' -f1)
    [ "$draft_weights_actual" = "$DRAFT_WEIGHTS_SHA256" ] \
      || die "draft weights do not match DRAFT_WEIGHTS_SHA256: actual $draft_weights_actual, pinned $DRAFT_WEIGHTS_SHA256 -- re-download the pinned revision or update the pin on BOTH nodes"
  else
    warn "DRAFT_WEIGHTS_SHA256 is unset: the two ranks cannot prove they hold the same draft; set it to: sha256sum $DRAFT_MODEL_HOST_PATH/*.safetensors"
  fi
fi

for name in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_CUDAGRAPH_CAPTURE_SIZE; do
  require_positive_integer "$name"
done
case "$NUM_SPECULATIVE_TOKENS" in
  ''|*[!0-9]*) die "NUM_SPECULATIVE_TOKENS must be a non-negative integer: $NUM_SPECULATIVE_TOKENS" ;;
esac
case "$ADAPTIVE_SPECULATIVE_TOKENS" in 0|1) : ;; *) die "ADAPTIVE_SPECULATIVE_TOKENS must be 0 or 1: $ADAPTIVE_SPECULATIVE_TOKENS" ;; esac
if [ "$ADAPTIVE_SPECULATIVE_TOKENS" = 1 ]; then
  [ "$SPECULATOR" != none ] && [ "$NUM_SPECULATIVE_TOKENS" -gt 0 ] \
    || die "ADAPTIVE_SPECULATIVE_TOKENS=1 needs a speculator with positive depth"
  if [ -z "$ADAPTIVE_SPECULATIVE_TOKENS_INITIAL" ]; then
    if [ "$NUM_SPECULATIVE_TOKENS" -lt 3 ]; then ADAPTIVE_SPECULATIVE_TOKENS_INITIAL=$NUM_SPECULATIVE_TOKENS
    else ADAPTIVE_SPECULATIVE_TOKENS_INITIAL=3; fi
  fi
  require_positive_integer ADAPTIVE_SPECULATIVE_TOKENS_INITIAL
  require_positive_integer ADAPTIVE_SPECULATIVE_TOKENS_WINDOW
  [ "$ADAPTIVE_SPECULATIVE_TOKENS_INITIAL" -le "$NUM_SPECULATIVE_TOKENS" ] \
    || die "ADAPTIVE_SPECULATIVE_TOKENS_INITIAL must not exceed NUM_SPECULATIVE_TOKENS"
fi
case "$MEM_PREFLIGHT" in die|warn|off) : ;; *) die "MEM_PREFLIGHT must be die, warn, or off: $MEM_PREFLIGHT" ;; esac
case "$FABRIC_PROFILE" in single|dual) : ;; *) die "FABRIC_PROFILE must be single or dual: $FABRIC_PROFILE" ;; esac
case "$GENERATION_CONFIG" in auto|vllm) : ;; *) die "GENERATION_CONFIG must be auto or vllm: $GENERATION_CONFIG" ;; esac
case "$COMPILATION_LEVEL" in ""|0|1|2|3) : ;; *) die "COMPILATION_LEVEL must be empty or 0-3: $COMPILATION_LEVEL" ;; esac
case "$INSTANTTENSOR_COPY" in
  auto) ;;
  0|1) warn "INSTANTTENSOR_COPY=$INSTANTTENSOR_COPY passes instanttensor_copy to the loader; pins >= 6575b5ac (2026-09-05) removed that option (use LOAD_FORMAT=fastsafetensors instead)" ;;
  *) die "INSTANTTENSOR_COPY must be auto, 0, or 1: $INSTANTTENSOR_COPY" ;;
esac
# The loader set is a property of the image, not the model.
case "$LOAD_FORMAT" in
  instanttensor|fastsafetensors|auto|safetensors) ;;
  b12x)
    case ",${VLLM_PLUGINS-}," in
      *,b12x_loader,*) ;;
      *) die "LOAD_FORMAT=b12x needs the loader plugin: set VLLM_PLUGINS=b12x_loader (vllm pin >= 9e500700, 2026-09-06)" ;;
    esac ;;
  *) die "LOAD_FORMAT must be instanttensor, fastsafetensors, b12x, auto, or safetensors: $LOAD_FORMAT" ;;
esac
case "$KDA_PREFILL_BACKEND" in ""|auto|triton|flashkda|b12x) : ;; *) die "KDA_PREFILL_BACKEND must be empty, auto, triton, flashkda, or b12x: $KDA_PREFILL_BACKEND" ;; esac
case "$GDN_DECODE_KERNEL" in ""|b12x|cuda|triton) : ;; *) die "GDN_DECODE_KERNEL must be empty, b12x, cuda, or triton: $GDN_DECODE_KERNEL" ;; esac
case "$FAIRNESS_ENGINE" in ""|compute_share|micro_slicing) : ;; *) die "FAIRNESS_ENGINE must be empty, compute_share, or micro_slicing: $FAIRNESS_ENGINE" ;; esac
case "$RECURRENT_CHECKPOINT_POLICY" in ""|auto|request_boundaries|aligned) : ;; *) die "RECURRENT_CHECKPOINT_POLICY must be empty, auto, request_boundaries, or aligned: $RECURRENT_CHECKPOINT_POLICY" ;; esac
if [ -n "$PREFILL_COMPUTE_SHARE" ]; then
  [ "$FAIRNESS_ENGINE" = compute_share ] || die "PREFILL_COMPUTE_SHARE requires FAIRNESS_ENGINE=compute_share"
  awk -v v="$PREFILL_COMPUTE_SHARE" 'BEGIN{ exit !(v+0 > 0 && v+0 < 1 && v ~ /^[0-9]*\.?[0-9]+$/) }' \
    || die "PREFILL_COMPUTE_SHARE must be a fraction in (0,1): $PREFILL_COMPUTE_SHARE"
fi
[ -z "$FAIRNESS_ENGINE" ] || [ -z "$PREFILL_SCHEDULE_INTERVAL" ] \
  || warn "FAIRNESS_ENGINE and PREFILL_SCHEDULE_INTERVAL are alternative answers to prefill/decode contention; the interval is normally 1 when a fairness engine is on"
for name in SAMPLING_TEMPERATURE SAMPLING_TOP_P SAMPLING_TOP_K SAMPLING_MIN_P SAMPLING_REPETITION_PENALTY; do
  v=${!name-}
  [ -z "$v" ] || awk -v v="$v" 'BEGIN{ exit !(v ~ /^[0-9]*\.?[0-9]+$/) }' || die "$name must be a number: $v"
done
require_positive_integer BLOCK_SIZE

# Multimodal: emitted only for a VL checkpoint. Every flag here costs memory
# that comes out of the same unified pool as KV, so none of it is emitted for
# a text-only model.
mm_args=()
if [ "$MODEL_MULTIMODAL" = 1 ]; then
  require_bool LANGUAGE_MODEL_ONLY
  require_bool LIMIT_MM
  case "$MM_PROCESSOR_CACHE_GB" in ""|[0-9]*) : ;; *) die "MM_PROCESSOR_CACHE_GB must be a number: $MM_PROCESSOR_CACHE_GB" ;; esac
  case "$MM_ENCODER_TP_MODE" in ""|data|weights) : ;; *) die "MM_ENCODER_TP_MODE must be empty, data, or weights: $MM_ENCODER_TP_MODE" ;; esac
  if [ "$LANGUAGE_MODEL_ONLY" = 1 ]; then
    mm_args=(--language-model-only)
  else
    if [ "$LIMIT_MM" = 1 ]; then
      # 0 is legitimate per modality: it disables that modality (MM_VIDEOS=0
      # drops the max-size video profile item, most of the vision reservation).
      require_nonnegative_integer MM_IMAGES
      require_nonnegative_integer MM_VIDEOS
      [ "$((MM_IMAGES + MM_VIDEOS))" -gt 0 ] \
        || warn "MM_IMAGES=0 and MM_VIDEOS=0 disable every modality; LANGUAGE_MODEL_ONLY=1 says that more clearly"
      mm_args=(--limit-mm-per-prompt "$(printf '{"image":%d,"video":%d}' "$MM_IMAGES" "$MM_VIDEOS")")
    else
      # Uncapped: the engine allows 999 items per modality per prompt, and in
      # profiling mode it sizes its dummy prompt from that limit.
      [ -n "$KV_CACHE_MEMORY_BYTES" ] \
        || warn "LIMIT_MM=0 without KV_CACHE_MEMORY_BYTES: memory profiling sizes its dummy prompt at the engine's 999-per-modality default"
    fi
    [ -z "$MM_PROCESSOR_CACHE_GB" ] || mm_args+=(--mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB")
    [ -z "$MM_ENCODER_TP_MODE" ] || mm_args+=(--mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")
  fi
else
  for name in MM_PROCESSOR_CACHE_GB MM_ENCODER_TP_MODE; do
    [ -z "${!name-}" ] || warn "$name is set but MODEL_MULTIMODAL=0; the flag is not emitted"
  done
fi
[ -z "$PREFILL_SCHEDULE_INTERVAL" ] || require_positive_integer PREFILL_SCHEDULE_INTERVAL
[ -z "$CONTAINER_MEMORY_GB" ] || require_positive_integer CONTAINER_MEMORY_GB
if [ -n "$CHAT_TEMPLATE_HOST_PATH" ]; then
  case "$CHAT_TEMPLATE_HOST_PATH" in /*) ;; *) die "CHAT_TEMPLATE_HOST_PATH must be absolute: $CHAT_TEMPLATE_HOST_PATH" ;; esac
  [ -f "$CHAT_TEMPLATE_HOST_PATH" ] || die "CHAT_TEMPLATE_HOST_PATH does not exist: $CHAT_TEMPLATE_HOST_PATH"
fi
if [ "$INSTANTTENSOR_COPY" != auto ] && [ "$LOAD_FORMAT" != instanttensor ]; then
  warn "INSTANTTENSOR_COPY=$INSTANTTENSOR_COPY has no effect at LOAD_FORMAT=$LOAD_FORMAT"
fi

if [ "$MAX_MODEL_LEN" != auto ]; then
  require_positive_integer MAX_MODEL_LEN
else
  warn "MAX_MODEL_LEN=auto resolves to the checkpoint maximum; on a Spark pair a 1M profile can starve the KV pool -- set the measured value"
fi

require_port API_PORT
require_port MASTER_PORT
[ "$API_PORT" != "$MASTER_PORT" ] || die "API_PORT and MASTER_PORT must differ"

require_unit_fraction GPU_MEMORY_UTILIZATION

# KV_CACHE_MEMORY_BYTES and GPU_MEMORY_UTILIZATION are alternatives: a pinned
# byte count makes vLLM skip profiling and ignore the fraction. Profile once
# (bytes empty), read the pool vLLM reports, then pin it. For hybrid models
# the state cache shares the reservation -- pin what vLLM reports, never a
# number carried over from another architecture.
if [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
  require_positive_integer KV_CACHE_MEMORY_BYTES
  if grep -qE '^[[:space:]]*GPU_MEMORY_UTILIZATION=' "$env_file"; then
    warn "KV_CACHE_MEMORY_BYTES is set; vLLM skips profiling and IGNORES GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
  fi
else
  # GB10 unified memory: ~121.7 GiB total, ~10.5 GiB held by the OS, so vLLM
  # refuses a utilization above roughly 0.913 before loading anything.
  awk -v v="$GPU_MEMORY_UTILIZATION" 'BEGIN{ exit !(v+0 > 0.90) }' \
    && warn "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION is at or past the GB10 free-memory ceiling (~0.913); vLLM refuses at startup when the request exceeds free memory"
fi

case "$ENABLE_FLASHINFER_AUTOTUNE" in ""|0|1) : ;; *) die "ENABLE_FLASHINFER_AUTOTUNE must be empty, 0, or 1: $ENABLE_FLASHINFER_AUTOTUNE" ;; esac
for name in ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL TRUST_REMOTE_CODE \
  VLLM_ENABLE_PCIE_ALLREDUCE; do
  require_bool "$name"
done

case " $KV_CACHE_DTYPE_CHOICES " in
  *" $KV_CACHE_DTYPE "*) ;;
  *) die "KV_CACHE_DTYPE must be one of: $KV_CACHE_DTYPE_CHOICES (got $KV_CACHE_DTYPE)" ;;
esac
case "$CUDAGRAPH_MODE" in
  FULL|FULL_AND_PIECEWISE|PIECEWISE|NONE) ;;
  *) die "CUDAGRAPH_MODE must be FULL, FULL_AND_PIECEWISE, PIECEWISE, or NONE: $CUDAGRAPH_MODE" ;;
esac

# GB10 is sm_121; sm_120a (RTX Pro 6000) CuTeDSL kernels build and then fail at load.
[ "$CUTE_DSL_ARCH" = sm_121a ] \
  || die "CUTE_DSL_ARCH must be sm_121a on DGX Spark (got $CUTE_DSL_ARCH); sm_120a is the RTX Pro 6000 value"

# The B12X PCIe allreduce is intra-node P2P. On a pair every TP allreduce
# crosses the CX7 link via NCCL; the image bakes 1 for single-node boxes.
[ "$VLLM_ENABLE_PCIE_ALLREDUCE" = 0 ] \
  || die "VLLM_ENABLE_PCIE_ALLREDUCE must be 0 on a two-node pair (TP allreduce goes over RoCE via NCCL)"

# One decode step is MAX_NUM_SEQS x (spec tokens + 1) tokens. Above the
# capture size the largest batches fall out of cudagraph replay; above the
# step budget vLLM cannot schedule a full step at all.
decode_batch=$(( MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1) ))
if [ "$decode_batch" -gt "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]; then
  warn "a full decode step is $decode_batch tokens (MAX_NUM_SEQS x (NUM_SPECULATIVE_TOKENS+1)) but MAX_CUDAGRAPH_CAPTURE_SIZE=$MAX_CUDAGRAPH_CAPTURE_SIZE; the largest batches fall out of cudagraph replay"
fi
if [ "$MAX_NUM_BATCHED_TOKENS" -lt "$decode_batch" ]; then
  die "MAX_NUM_BATCHED_TOKENS ($MAX_NUM_BATCHED_TOKENS) is below one speculative decode step ($decode_batch)"
fi
if [ "$ENABLE_CHUNKED_PREFILL" = 1 ] && [ "$MAX_NUM_BATCHED_TOKENS" -lt 2048 ]; then
  warn "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS: long prefills will take very many chunks. Raising it trades a larger transient activation peak (unified memory, so it comes out of the KV pool) for prefill throughput -- measure both."
fi

model_validate   # hook: model-specific keys and known-bad combinations

# ------------------------------------------------------------ fabric checks
# Launcher-enforced constants for a direct-cable RoCEv2 pair. Change one only
# with a reason better than the message that refuses it.

[ "$NCCL_SOCKET_IFNAME" = "$GLOO_SOCKET_IFNAME" ] \
  || die "NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME must match on a pair"
[ "$NCCL_NET" = IB ] || die "NCCL_NET must be IB (RoCE presents IB semantics over Ethernet)"
[ "$NCCL_NET_PLUGIN" = none ] || die "NCCL_NET_PLUGIN must be none"
[ "$NCCL_IB_DISABLE" = 0 ] || die "NCCL_IB_DISABLE must be 0 (the image bakes 1 for single-node use; override it in the env file or the pair silently falls back to TCP)"
# FABRIC_PROFILE selects the cabling, and the launcher then enforces every
# knob that follows from it. It is named explicitly rather than inferred from
# NCCL_IB_MERGE_NICS so that a stray edit or a copied env file cannot put a
# single-cable pair onto a profile its hardware does not have.
#
#   single (default): ONE cable between the nodes. MERGE_NICS=0,
#     SUBNET_AWARE_ROUTING=0, exactly one device in NCCL_IB_HCA. Nothing to
#     route around and nothing to merge; anything else is a misconfiguration.
#   dual: BOTH CX7 ports cabled directly node-to-node. MERGE_NICS=1,
#     SUBNET_AWARE_ROUTING=1, both rails in NCCL_IB_HCA -- NCCL merges the
#     links for roughly 2x cross-node bandwidth. Do not select this without
#     the second cable physically present: NCCL will advertise a rail that
#     cannot carry traffic and the pair hangs during the first allreduce.
if [ "$FABRIC_PROFILE" = dual ]; then
  [ "$NCCL_IB_MERGE_NICS" = 1 ] \
    || die "FABRIC_PROFILE=dual requires NCCL_IB_MERGE_NICS=1"
  [ "$NCCL_IB_SUBNET_AWARE_ROUTING" = 1 ] \
    || die "FABRIC_PROFILE=dual requires NCCL_IB_SUBNET_AWARE_ROUTING=1"
  case "$NCCL_IB_HCA" in
    *,*) ;;
    *) die "FABRIC_PROFILE=dual needs both rails in NCCL_IB_HCA (names from ibv_devices), got: $NCCL_IB_HCA" ;;
  esac
else
  [ "$NCCL_IB_MERGE_NICS" = 0 ] \
    || die "single-rail pair must set NCCL_IB_MERGE_NICS=0 (FABRIC_PROFILE=dual only with both ports cabled)"
  [ "$NCCL_IB_SUBNET_AWARE_ROUTING" = 0 ] \
    || warn "NCCL_IB_SUBNET_AWARE_ROUTING=$NCCL_IB_SUBNET_AWARE_ROUTING on a single-rail pair: off the validated 0/0 profile. Upstream's single-rail Spark launcher runs 1/0, so it is allowed -- but unmeasured here"
fi
[ "$NCCL_PROTO" = LL,LL128,Simple ] || die "NCCL_PROTO must be LL,LL128,Simple"
[ "$NCCL_P2P_LEVEL" = SYS ] || die "NCCL_P2P_LEVEL must be SYS"
[ "$NCCL_CROSS_NIC" = 1 ] || die "NCCL_CROSS_NIC must be 1"
[ "$NCCL_CUMEM_ENABLE" = 0 ] || die "NCCL_CUMEM_ENABLE must be 0 (conflicts with expandable_segments)"
[ "$NCCL_IGNORE_CPU_AFFINITY" = 1 ] || die "NCCL_IGNORE_CPU_AFFINITY must be 1 (Grace unified layout defeats NUMA guesses)"

case ":$LD_PRELOAD:" in
  *":$VLLM_NCCL_SO_PATH:"*) ;;
  *) die "LD_PRELOAD must include VLLM_NCCL_SO_PATH ($VLLM_NCCL_SO_PATH)" ;;
esac

[ "$NODE_RANK" != 0 ] || [ "$MASTER_ADDR" = "$VLLM_HOST_IP" ] \
  || die "rank-0 MASTER_ADDR must equal rank-0 VLLM_HOST_IP"
[ "$NODE_RANK" != 1 ] || [ "$MASTER_ADDR" != "$VLLM_HOST_IP" ] \
  || die "rank-1 VLLM_HOST_IP must differ from MASTER_ADDR"

case "$NCCL_IB_HCA" in
  *,*) [ "$FABRIC_PROFILE" = dual ] \
         || die "a single-rail pair names exactly one RoCE device; got $NCCL_IB_HCA (set FABRIC_PROFILE=dual only if both ports are cabled)" ;;
esac
case "$NCCL_IB_GID_INDEX" in
  ''|*[!0-9]*) die "NCCL_IB_GID_INDEX must be a decimal integer" ;;
esac

# RoCEnante: b12x one-shot RoCE collectives for small TP all-reduces and
# all-gathers, with NCCL keeping everything above the cutoffs. Pair-level, so
# it is checked here; the values themselves ride to the container via
# --env-file. The proxy compiles at first boot into B12X_ROCE_CACHE_DIR.
if [ "${VLLM_ENABLE_ROCE_ALLREDUCE:-0}" = 1 ]; then
  for name in VLLM_ROCE_ALLREDUCE_MAX_SIZE VLLM_ROCE_ALLGATHER_MAX_SIZE; do
    [ -z "${!name-}" ] || require_positive_integer "$name"
  done
  case "${B12X_ROCE_CACHE_DIR-}" in
    "") warn "VLLM_ENABLE_ROCE_ALLREDUCE=1 without B12X_ROCE_CACHE_DIR: the proxy recompiles into the image default every boot; point it under /cache" ;;
    /cache*) ;;
    *) warn "B12X_ROCE_CACHE_DIR=$B12X_ROCE_CACHE_DIR is outside the mounted /cache; the compiled proxy will not survive the container" ;;
  esac
  # A reduce of one prefill chunk is MAX_NUM_BATCHED_TOKENS x hidden x dtype
  # bytes per layer; a cutoff below that leaves prefill entirely on NCCL.
  [ -z "${VLLM_ROCE_ALLREDUCE_MAX_SIZE-}" ] || [ "$VLLM_ROCE_ALLREDUCE_MAX_SIZE" -ge 16777216 ] \
    || note_roce=1
  [ "${note_roce:-0}" = 0 ] \
    || warn "VLLM_ROCE_ALLREDUCE_MAX_SIZE=$VLLM_ROCE_ALLREDUCE_MAX_SIZE is below the 16 MiB that covers a 4096-token prefill chunk's per-layer reduce on a 2048-wide model; prefill stays on NCCL"
fi

# MASTER_ADDR must be on a directly connected subnet. A one-digit typo routes
# to the internet and the torch.distributed rendezvous hangs forever silently.
if command -v ip >/dev/null 2>&1; then
  master_route="$(ip route get "$MASTER_ADDR" 2>/dev/null | head -1)"
  case "$master_route" in
    "")        die "MASTER_ADDR=$MASTER_ADDR has no route from this host -- typo?" ;;
    *" via "*) die "MASTER_ADDR=$MASTER_ADDR is not on a directly connected subnet (route: $master_route) -- it must be rank 0's address on the CX7 link" ;;
  esac
fi

# ------------------------------------------------------- host / image checks

# Wrong rank's env file on a host: both ranks fight over ports under --network host.
if command -v ip >/dev/null 2>&1; then
  if ! ip -4 -o addr show 2>/dev/null | grep -qw "$VLLM_HOST_IP"; then
    die "VLLM_HOST_IP ($VLLM_HOST_IP) is not an IPv4 address on this host; wrong rank's env file?"
  fi
fi

# --network host: an occupied port is an immediate bind failure at exec.
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

# Memory preflight (profiling mode only). On Grace unified memory CUDA free
# memory tracks MemFree, so page cache left by a previous model load counts
# AGAINST it and vLLM fails its own check before loading anything. Name the
# culprit here: stale container, reclaimable page cache, or a real consumer.
if [ "$MEM_PREFLIGHT" != off ] && [ -r /proc/meminfo ] && [ -z "$KV_CACHE_MEMORY_BYTES" ]; then
  mem_total_kib=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
  mem_free_kib=$(awk '/^MemFree:/{print $2}' /proc/meminfo)
  mem_avail_kib=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  required_kib=$(awk -v t="$mem_total_kib" -v u="$GPU_MEMORY_UTILIZATION" 'BEGIN{printf "%d", t*u}')
  if [ "$mem_free_kib" -lt "$required_kib" ]; then
    deficit_gib=$(awk -v r="$required_kib" -v f="$mem_free_kib" 'BEGIN{printf "%.1f", (r-f)/1048576}')
    stale="$("$CONTAINER_RUNTIME" ps --format '{{.Names}}' 2>/dev/null | grep -c "^${MODEL_ID}-r" || true)"
    if [ "${stale:-0}" -gt 0 ]; then
      die "only $(awk -v f="$mem_free_kib" 'BEGIN{printf "%.1f", f/1048576}') GiB free but utilization $GPU_MEMORY_UTILIZATION needs ~$(awk -v r="$required_kib" 'BEGIN{printf "%.1f", r/1048576}') GiB (short $deficit_gib GiB) -- a ${MODEL_ID} container is already running on this host; --down it first"
    elif [ "$mem_avail_kib" -ge "$required_kib" ]; then
      # The node LOOKS idle on any dashboard that reports total-minus-available,
      # but vLLM's startup gate reads CUDA free memory, which tracks MemFree.
      cache_gib=$(awk -v a="$mem_avail_kib" -v f="$mem_free_kib" 'BEGIN{printf "%.1f", (a-f)/1048576}')
      msg="MemFree=$(awk -v f="$mem_free_kib" 'BEGIN{printf "%.1f", f/1048576}') GiB is $deficit_gib GiB short of utilization $GPU_MEMORY_UTILIZATION, but MemAvailable=$(awk -v a="$mem_avail_kib" 'BEGIN{printf "%.1f", a/1048576}') GiB suffices: ~$cache_gib GiB of reclaimable page cache (a previous model load) is counting against CUDA free memory. Reclaim: sync && echo 3 | sudo tee /proc/sys/vm/drop_caches -- then relaunch (MEM_PREFLIGHT=warn to proceed anyway)"
      if [ "$MEM_PREFLIGHT" = warn ]; then warn "$msg"; else die "$msg"; fi
    else
      die "this host has only $(awk -v a="$mem_avail_kib" 'BEGIN{printf "%.1f", a/1048576}') GiB available; utilization $GPU_MEMORY_UTILIZATION needs ~$(awk -v r="$required_kib" 'BEGIN{printf "%.1f", r/1048576}') GiB. Find the consumer (docker ps, top) or lower GPU_MEMORY_UTILIZATION"
    fi
  fi
fi

# --device /dev/infiniband fails at docker run with an error naming the path, not the cause.
[ -e /dev/infiniband ] \
  || die "/dev/infiniband does not exist; the RDMA stack is not up (check: lsmod | grep mlx5_ib)"

# --ipc host makes the HOST's /dev/shm authoritative; --shm-size is inert.
if [ -d /dev/shm ]; then
  shm_mb=$(df -Pm /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
  case "$shm_mb" in
    ''|*[!0-9]*) ;;
    *) [ "$shm_mb" -ge 8192 ] \
         || warn "/dev/shm is ${shm_mb} MiB; --ipc host makes this the real limit and vLLM's shm_broadcast can stall below ~8 GiB" ;;
  esac
fi

# LD_PRELOAD is the most fragile line: a path absent from the image kills
# EVERY process at exec with an opaque loader error (or later with
# `undefined symbol: cuTensorMapEncodeTiled` when the compat shim is the one
# missing). Verify inside the image itself.
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

speculative_args=()
if [ "$SPECULATOR" != none ] && [ "$NUM_SPECULATIVE_TOKENS" -gt 0 ]; then
  speculative_args=(--speculative-config "$(model_speculative_config)")   # hook
fi

compilation_config=$(printf '{"cudagraph_mode":"%s","custom_ops":["all"]}' "$CUDAGRAPH_MODE")

# Optional serve flags. Each is omitted unless configured, because an unknown
# CLI flag is a hard argparse failure at exec on an image that predates it.
chat_template_container_path=/models/chat_template.jinja
extra_args=()
[ -z "$PREFILL_SCHEDULE_INTERVAL" ] || extra_args+=(--prefill-schedule-interval "$PREFILL_SCHEDULE_INTERVAL")
case "$INSTANTTENSOR_COPY" in
  auto) ;;
  0) extra_args+=(--model-loader-extra-config '{"instanttensor_copy":false}') ;;
  1) extra_args+=(--model-loader-extra-config '{"instanttensor_copy":true}') ;;
esac
[ -z "$COMPILATION_LEVEL" ] || extra_args+=("-O$COMPILATION_LEVEL")
[ "$GENERATION_CONFIG" = auto ] || extra_args+=(--generation-config "$GENERATION_CONFIG")
[ -z "$CHAT_TEMPLATE_HOST_PATH" ] || extra_args+=(--chat-template "$chat_template_container_path")
[ -z "$FAIRNESS_ENGINE" ] || extra_args+=(--fairness-engine "$FAIRNESS_ENGINE")
[ -z "$PREFILL_COMPUTE_SHARE" ] || extra_args+=(--prefill-compute-share "$PREFILL_COMPUTE_SHARE")
[ -z "$RECURRENT_CHECKPOINT_POLICY" ] || extra_args+=(--recurrent-checkpoint-policy "$RECURRENT_CHECKPOINT_POLICY")
[ -z "$KDA_PREFILL_BACKEND" ] || extra_args+=(--kda-prefill-backend "$KDA_PREFILL_BACKEND")
[ -z "$GDN_DECODE_KERNEL" ] || extra_args+=(--gdn-decode-kernel "$GDN_DECODE_KERNEL")
[ -z "$API_KEY" ] || extra_args+=(--api-key "$API_KEY")

# Sampling and chat-template defaults are assembled HERE: JSON in a sourced
# env file loses its quotes, so the env file carries scalars only.
sampling_json=""
for _f in temperature:SAMPLING_TEMPERATURE top_p:SAMPLING_TOP_P top_k:SAMPLING_TOP_K \
          min_p:SAMPLING_MIN_P repetition_penalty:SAMPLING_REPETITION_PENALTY; do
  _key=${_f%%:*}; _var=${_f##*:}; _val=${!_var-}
  [ -n "$_val" ] || continue
  sampling_json="${sampling_json:+$sampling_json,}\"$_key\":$_val"
done
[ -z "$sampling_json" ] || extra_args+=(--override-generation-config "{$sampling_json}")
template_json="$(model_chat_template_kwargs)"   # hook
[ -z "$template_json" ] || extra_args+=(--default-chat-template-kwargs "{$template_json}")

case "$CONTAINER_RUNTIME" in
  podman) gpu_args=(--device nvidia.com/gpu=all --security-opt label=disable) ;;
  *)      gpu_args=(--gpus all) ;;
esac

mount_args=(
  -v "$MODEL_HOST_PATH:$MODEL_CONTAINER_PATH:ro"
  -v "$CACHE_HOST_PATH:/cache"
)
if model_needs_draft; then
  mount_args+=(-v "$DRAFT_MODEL_HOST_PATH:$DRAFT_CONTAINER_PATH:ro")
fi
if [ -n "$CHAT_TEMPLATE_HOST_PATH" ]; then
  mount_args+=(-v "$CHAT_TEMPLATE_HOST_PATH:$chat_template_container_path:ro")
fi
if [ -n "$TORCH_PROFILE_HOST_DIR" ]; then
  mkdir -p "$TORCH_PROFILE_HOST_DIR"
  mount_args+=(-v "$TORCH_PROFILE_HOST_DIR:/profiles")
fi

# Runtime flags built conditionally rather than via ${VAR:+...} inside the
# command array, which would depend on word splitting to produce two flags.
runtime_args=()
if [ -n "$CONTAINER_MEMORY_GB" ]; then
  # GB10 unified memory: this cgroup cap counts the GPU-visible allocation too.
  # Set it above the model footprint plus the KV reservation or the load is
  # OOM-killed by the kernel, not by vLLM.
  runtime_args+=(--memory "${CONTAINER_MEMORY_GB}g" --memory-swap "$((CONTAINER_MEMORY_GB + 4))g")
fi

# Explicit -e beats both --env-file and the image's baked ENV.
container_env=(
  # Pre-rename baked value `cpp` crashes eager env validation on current vLLM
  # even with the PCIe path disabled. No-op on builder-patched images.
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
)
# The INSTANTTENSOR_* bounds are read by the instanttensor library only; under
# any other loader they are inert, so they are passed only when it is selected.
if [ "$LOAD_FORMAT" = instanttensor ]; then
  container_env+=(
    -e INSTANTTENSOR_BUFFER_SIZE="$INSTANTTENSOR_BUFFER_SIZE"
    -e INSTANTTENSOR_IO_DEPTH="$INSTANTTENSOR_IO_DEPTH"
    -e INSTANTTENSOR_CONCURRENCY="$INSTANTTENSOR_CONCURRENCY"
    -e INSTANTTENSOR_CHUNK_SIZE="$INSTANTTENSOR_CHUNK_SIZE"
  )
fi
if [ -n "$TORCH_PROFILE_HOST_DIR" ]; then
  container_env+=(-e VLLM_TORCH_PROFILER_DIR=/profiles)
fi
model_container_env   # hook

# Comma-separated alias list; vLLM answers to every name and reports the first.
IFS=',' read -r -a served_names <<< "$SERVED_MODEL_NAME"
for served_name in "${served_names[@]}"; do
  [ -n "$served_name" ] || die "SERVED_MODEL_NAME has an empty entry (stray comma): $SERVED_MODEL_NAME"
done

model_args=()
model_serve_args   # hook

command=(
  "$CONTAINER_RUNTIME" run -d
  --name "$container_name"
  --pull never                       # the image must already be on this node (docker save|load)
  --network host                     # RoCE + rendezvous need the host's real addresses
  --ipc host                         # host /dev/shm for vLLM's shm_broadcast
  --shm-size "$SHM_SIZE"
  "${gpu_args[@]}"
  --ulimit memlock=-1:-1             # RDMA registration needs unlimited pinned memory
  "${runtime_args[@]}"
  --device /dev/infiniband
  "${mount_args[@]}"
  --env-file "$env_file"
  "${container_env[@]}"
  --entrypoint /opt/venv/bin/vllm
  "$SERVING_IMAGE"
  serve "$MODEL_CONTAINER_PATH"

  # --- topology (pair-fixed) ----------------------------------------------
  --tensor-parallel-size 2
  --nnodes 2
  --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --distributed-executor-backend mp
  --pipeline-parallel-size 1
  --decode-context-parallel-size 1

  # --- memory --------------------------------------------------------------
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --block-size "$BLOCK_SIZE"         # public block-table unit; hybrids override the physical page

  # --- kernels / graphs ----------------------------------------------------
  --attention-backend "$ATTENTION_BACKEND"
  --moe-backend "$MOE_BACKEND"
  --load-format "$LOAD_FORMAT"
  --compilation-config "$compilation_config"
  --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"

  # --- scheduling ----------------------------------------------------------
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"

  # --- optional / multimodal / model-specific ------------------------------
  "${mm_args[@]}"
  "${extra_args[@]}"
  "${model_args[@]}"
  "${speculative_args[@]}"
  --served-model-name "${served_names[@]}"

  # --- observability -------------------------------------------------------
  --enable-prompt-tokens-details
  --enable-force-include-usage
  --enable-request-id-headers
)

# `if` not `&&`: a disabled toggle must not trip set -e.
if [ "$TRUST_REMOTE_CODE" = 1 ]; then command+=(--trust-remote-code); fi
if [ "$ENABLE_PREFIX_CACHING" = 1 ]; then command+=(--enable-prefix-caching); fi
if [ "$ENABLE_CHUNKED_PREFILL" = 1 ]; then command+=(--enable-chunked-prefill); fi
case "$ENABLE_FLASHINFER_AUTOTUNE" in
  1) command+=(--enable-flashinfer-autotune) ;;
  0) command+=(--no-enable-flashinfer-autotune) ;;
esac
if [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
  command+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi

if [ "$NODE_RANK" = 0 ]; then
  command+=(--host 0.0.0.0 --port "$API_PORT")
else
  command+=(--headless)
fi

[ "${#passthrough[@]}" -eq 0 ] || command+=("${passthrough[@]}")

# ------------------------------------------------------------------ output

printf "Local rank input checks passed.\n"
printf '  rank:                    %s\n' "$NODE_RANK"
printf '  runtime:                 %s\n' "$CONTAINER_RUNTIME"
printf '  image:                   %s\n' "$SERVING_IMAGE"
printf '  model:                   %s\n' "$MODEL_HOST_PATH"
if model_needs_draft; then
  printf '  draft model:             %s\n' "$DRAFT_MODEL_HOST_PATH"
  printf '  draft weights sha256:    %s\n' "${DRAFT_WEIGHTS_SHA256:-UNPINNED}"
fi
printf '  served names:            %s\n' "$SERVED_MODEL_NAME"
printf '  cache:                   %s\n' "$CACHE_HOST_PATH"
printf '  MAX_MODEL_LEN:           %s\n' "$MAX_MODEL_LEN"
printf '  MAX_NUM_SEQS:            %s\n' "$MAX_NUM_SEQS"
printf '  MAX_NUM_BATCHED_TOKENS:  %s\n' "$MAX_NUM_BATCHED_TOKENS"
printf '  SPECULATOR:              %s (%s draft tokens%s)\n' "$SPECULATOR" "$NUM_SPECULATIVE_TOKENS" \
  "$([ "$ADAPTIVE_SPECULATIVE_TOKENS" = 1 ] && printf ', adaptive from %s' "$ADAPTIVE_SPECULATIVE_TOKENS_INITIAL")"
printf '  KV_CACHE_MEMORY_BYTES:   %s (%s)\n' \
  "${KV_CACHE_MEMORY_BYTES:-profiled at $GPU_MEMORY_UTILIZATION}" "$KV_CACHE_DTYPE"
if [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
  printf '  GPU_MEMORY_UTILIZATION:  %s (IGNORED - bytes are pinned)\n' "$GPU_MEMORY_UTILIZATION"
else
  printf '  GPU_MEMORY_UTILIZATION:  %s\n' "$GPU_MEMORY_UTILIZATION"
fi
printf '  ATTENTION/MOE backend:   %s / %s\n' "$ATTENTION_BACKEND" "$MOE_BACKEND"
printf '  BLOCK_SIZE:              %s\n' "$BLOCK_SIZE"
if [ "$MODEL_MULTIMODAL" = 1 ]; then
  printf '  multimodal:              %s\n' \
    "$([ "$LANGUAGE_MODEL_ONLY" = 1 ] && echo 'text-only (--language-model-only)' \
       || printf 'images=%s videos=%s%s' "$MM_IMAGES" "$MM_VIDEOS" "$([ "$LIMIT_MM" = 1 ] || echo ' (UNCAPPED)')")"
fi
printf '  RoCEnante:               %s\n' \
  "$([ "${VLLM_ENABLE_ROCE_ALLREDUCE:-0}" = 1 ] && echo "on (<= ${VLLM_ROCE_ALLREDUCE_MAX_SIZE:-default} B)" || echo 'off (NCCL for all collectives)')"
printf '  CUDAGRAPH_MODE:          %s (capture %s)\n' "$CUDAGRAPH_MODE" "$MAX_CUDAGRAPH_CAPTURE_SIZE"
printf '  LOAD_FORMAT:             %s\n' "$LOAD_FORMAT"
printf '  fabric profile:          %s-rail (%s)\n' "$FABRIC_PROFILE" "$NCCL_IB_HCA"
[ "${#extra_args[@]}" -eq 0 ] || printf '  optional flags:          %s\n' "${extra_args[*]}"
[ -z "$CONTAINER_MEMORY_GB" ] || printf '  container memory cap:    %s GiB\n' "$CONTAINER_MEMORY_GB"
[ -z "$TORCH_PROFILE_HOST_DIR" ] || printf '  torch profiler dir:      %s\n' "$TORCH_PROFILE_HOST_DIR"
model_summary   # hook
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

exec "${command[@]}"
