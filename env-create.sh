#!/usr/bin/env bash
# env-create.sh — interactively fill a model's rank env file from its
# -rank.env.example, using a per-model profile under profiles/.
#
# Usage:  ./env-create.sh
#   1. pick the model directory
#   2. (optionally) back up existing rank-*.env
#   3. pick rank 0 (API node) or 1 (headless worker)
#   4. answer the pair-wide fabric identity prompts (defaults from profiles/base.env)
#   5. answer / reuse the model's profile values (model + cache paths, serving
#      image, API key, API port, served model name)
#   6. writes deployments/<model>/rank-<rank>.env and runs the model's
#      pair launcher --check on it when one exists.
#
# Profiles live at deployments/profiles/<env-stem>-profile.env where
# <env-stem> is the env example basename minus "-rank.env.example".

set -euo pipefail

deployments_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profiles_dir="$deployments_root/profiles"
base_env="$profiles_dir/base.env"

die() { printf 'env-create: %s\n' "$*" >&2; exit 20; }

# Strip inline comments and trailing whitespace from a KEY=value line.
# Bare `read` with IFS= only keeps leading whitespace: value-safe for paths.
parse_kv() { # $1 = line -> sets KEY, VAL
  KEY=${1%%=*}
  VAL=${1#*=}
  VAL=${VAL%%#*}          # drop trailing comment
  # trim surrounding whitespace
  VAL=${VAL#"${VAL%%[![:space:]]*}"}
  VAL=${VAL%"${VAL##*[![:space:]]}"}
}

# Read one KEY=value line set from a file, in file order, printing each pair
# as `KEY<TAB>value`. Non-comment, non-empty lines with an `=` only.
read_kv_pairs() { # $1 = file
  local line
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%%#*}
    line=${line%"${line##*[![:space:]]}"}
    [[ $line == *=* && $line != =* ]] || continue
    parse_kv "$line"
    printf '%s\t%s\n' "$KEY" "$VAL"
  done < "$1"
}

# Prompt loop: read a non-empty answer into REPLY-compatible global $ans.
ask() { # $1 = prompt, $2 = default (optional, shown as [default])
  local prompt=$1 def=${2:-} input
  while :; do
    if [[ -n $def ]]; then
      printf '%s [%s]: ' "$prompt" "$def"
    else
      printf '%s: ' "$prompt"
    fi
    IFS= read -r input || die "stdin closed while prompting"
    input=${input%"${input##*[![:space:]]}"}
    if [[ -z $input && -n $def ]]; then input=$def; fi
    if [[ -n $input ]]; then ans=$input; return; fi
    printf '  a value is required\n' >&2
  done
}

ask_yn() { # $1 = question -> sets yn in {y,n}
  local input
  while :; do
    printf '%s [y/n]: ' "$1"
    IFS= read -r input || die "stdin closed while prompting"
    case ${input,,} in
      y|yes) yn=y; return ;;
      n|no)  yn=n; return ;;
    esac
    printf '  answer y or n\n' >&2
  done
}

ask_opt() { # $1 = prompt, $2 = default (optional); empty answer allowed -> $ans may be empty
  local prompt=$1 def=${2:-} input
  if [[ -n $def ]]; then
    printf '%s [%s]: ' "$prompt" "$def"
  else
    printf '%s: ' "$prompt"
  fi
  IFS= read -r input || die "stdin closed while prompting"
  input=${input%"${input##*[![:space:]]}"}
  [[ -z $input && -n $def ]] && input=$def
  ans=$input
}

ask_path() { # $1 = prompt, $2 = default; TAB completes paths when stdin is a terminal (readline)
  local prompt=$1 def=${2:-} input pstr
  if [[ -n $def ]]; then
    pstr="$prompt [$def]: "
  else
    pstr="$prompt: "
  fi
  while :; do
    if [[ -t 0 ]]; then
      IFS= read -r -e -p "$pstr" input || die "stdin closed while prompting"
    else
      printf '%s' "$pstr" >&2
      IFS= read -r input || die "stdin closed while prompting"
    fi
    input=${input%"${input##*[![:space:]]}"}
    if [[ -z $input && -n $def ]]; then input=$def; fi
    if [[ -n $input ]]; then ans=$input; return; fi
    printf '  a value is required\n' >&2
  done
}

# First value of a key in the chosen env example ("" if the line is absent or empty).
example_value() { # $1 = key
  local k v
  while IFS=$'\t' read -r k v; do
    if [[ $k == "$1" ]]; then printf '%s' "$v"; return; fi
  done < <(read_kv_pairs "$env_example")
}

# Pick one item from a numbered list -> sets pick (1-based index).
pick_from_list() { # $1 = question, rest = items
  local question=$1; shift
  local i input
  while :; do
    for i in "${!ITEMS[@]}"; do
      printf '  %d) %s\n' $((i + 1)) "${ITEMS[i]}"
    done
    printf '%s [1-%d]: ' "$question" "${#ITEMS[@]}"
    IFS= read -r input || die "stdin closed while prompting"
    if [[ $input =~ ^[0-9]+$ ]] && (( input >= 1 && input <= ${#ITEMS[@]} )); then
      pick=$input; return
    fi
    printf '  enter a number between 1 and %d\n' "${#ITEMS[@]}" >&2
  done
}

apply_kv() { # $1 = mode (skip|append), $2 = file rewritten in place
  local mode=$1 file=$2 tmp
  tmp=$(mktemp) || die "mktemp failed"
  awk -F'\t' -v mode="$mode" -v fname="$file" '
    FILENAME == "-" {
      if (!($1 in k)) order[++n] = $1
      k[$1] = $2
      next
    }
    {
      for (i = 1; i <= n; i++) {
        key = order[i]
        if ($0 ~ "^[[:space:]]*" key "[[:space:]]*=") {
          print key "=" k[key]; seen[key] = 1; next
        }
      }
      print
    }
    END {
      for (i = 1; i <= n; i++) {
        key = order[i]
        if (!(key in seen)) {
          if (mode == "append") print key "=" k[key]
          else print "env-create: note: no " key "= line in " fname "; value skipped" > "/dev/stderr"
        }
      }
    }
  ' - "$file" <&0 > "$tmp" && mv "$tmp" "$file"
}

# --- 1. model menu -----------------------------------------------------------
shopt -s nullglob
models=()
for d in "$deployments_root"/*/; do
  b=$(basename "$d")
  [[ $b == templates || $b == profiles ]] && continue
  models+=("$b")
done
shopt -u nullglob
(( ${#models[@]} > 0 )) || die "no model directories under $deployments_root"

ITEMS=("${models[@]}")
pick_from_list "Which model directory?" "${ITEMS[@]}"
model=${ITEMS[pick - 1]}
model_dir=$deployments_root/$model
env_examples=("$model_dir"/*_rank-env.example)
pair-serves=("$model_dir"/*_pair-serve.sh)
shopt -u nullglob
(( ${#env_examples[@]} > 0 )) || die "no *-rank.env.example in $model_dir"
if (( ${#env_examples[@]} == 1 )); then
  env_example=${env_examples[0]}
else
  ITEMS=("${env_examples[@]#$model_dir/}")
  pick_from_list "Which env example?" "${ITEMS[@]}"
  env_example=${env_examples[pick - 1]}
fi
stem=${env_example##*/}
stem=${stem%-rank.env.example}

pair-serve=
if (( ${#pair-serves[@]} == 1 )); then
  pair-serve=${pair-serves[0]}
elif (( ${#pair-serves[@]} > 1 )); then
  ITEMS=("${pair-serves[@]#$model_dir/}")
  pick_from_list "Which pair serve (for the post-write check)?" "${ITEMS[@]}"
  pair-serve=${pair-serves[pick - 1]}
else
  printf 'env-create: warning: no *_pair-serve.sh in %s — skipping post-check\n' "$model_dir" >&2
fi

# --- 3. backup prompt --------------------------------------------------------
timestamp=$(date +%Y%m%d-%H%M%S)
existing_envs=()
for f in "$model_dir"/rank-0.env "$model_dir"/rank-1.env; do
  [[ -f $f ]] && existing_envs+=("$f")
done
if (( ${#existing_envs[@]} > 0 )); then
  printf 'existing: %s\n' "${existing_envs[*]#$model_dir/}"
  ask_yn "Back up existing rank env file(s) before overwrite?"
  if [[ $yn == y ]]; then
    for f in "${existing_envs[@]}"; do
      mv "$f" "$model_dir/${timestamp}-${f##*/}"
      printf 'backed up: %s\n' "${f##*/} -> ${timestamp}-${f##*/}"
    done
  fi
fi

# --- 4. rank question --------------------------------------------------------
while :; do
  printf 'Rank? 0 = API node, 1 = headless worker [0/1]: '
  IFS= read -r ans || die "stdin closed while prompting"
  [[ $ans == 0 || $ans == 1 ]] && break
  printf '  enter 0 or 1\n' >&2
done
rank=$ans

# --- 5. base prompts (fabric identity) --------------------------------------
[[ -f $base_env ]] || die "missing $base_env (create it next to profiles/)"

declare -A base_vals=()
base_keys=()
while IFS=$'\t' read -r k v; do
  base_keys+=("$k")
  base_vals[$k]=$v
done < <(read_kv_pairs "$base_env")

answers=()
answer_key() { # $1 = key, $2 = prompt, $3 = default (optional)
  local key=$1 prompt=$2 def=${3:-}
  # non-empty, non-placeholder stored value -> editable default
  if [[ -z $def && -n ${base_vals[$key]:-} && ! ${base_vals[$key]} =~ ^\<.*\>$ ]]; then
    def=${base_vals[$key]}
  fi
  ask "$prompt" "$def"
  answers+=("$key"$'\t'"$ans")
}
for key in "${base_keys[@]}"; do
  case $key in
    MASTER_ADDR)         answer_key "$key" "Rank 0 fabric IPv4 (MASTER_ADDR)" ;;
    VLLM_HOST_IP_RANK1)  [[ $rank == 1 ]] && answer_key "$key" "Rank 1 fabric IPv4 (VLLM_HOST_IP)" ;;
    NCCL_SOCKET_IFNAME)  answer_key "$key" "Fabric interface name (NCCL_SOCKET_IFNAME)" ; nccl_ans=$ans ;;
    GLOO_SOCKET_IFNAME)  answer_key "$key" "Fabric interface name (GLOO_SOCKET_IFNAME)" "${nccl_ans:-}" ;;
    NCCL_IB_HCA)         answer_key "$key" "RDMA device (NCCL_IB_HCA)" ;;
    NCCL_IB_GID_INDEX)   answer_key "$key" "RoCEv2 IPv4 GID index (NCCL_IB_GID_INDEX)" ;;
    *)                   answer_key "$key" "Enter value for $key" ;;
  esac
done

# MASTER_ADDR also fills rank 0's VLLM_HOST_IP; GLOO default is the NCCL answer.
master_addr=
nccl_if=
gloo_if=
rank1_ip=
for pair in "${answers[@]}"; do
  KEY=${pair%%$'\t'*}
  VAL=${pair#*$'\t'}
  case $KEY in
    MASTER_ADDR)        master_addr=$VAL ;;
    VLLM_HOST_IP_RANK1) rank1_ip=$VAL ;;
    NCCL_SOCKET_IFNAME) nccl_if=$VAL ;;
    GLOO_SOCKET_IFNAME) gloo_if=$VAL ;;
  esac
done

# --- 6. profile resolution ---------------------------------------------------
profile=$profiles_dir/$stem-profile.env

# Prompt the model's six values (paths + image + API key + port + served name),
# honoring defaults from a seed source (an existing profile or rank-#.env),
# with API_PORT / SERVED_MODEL_NAME falling back to the env example.
# SERVED_MODEL_NAME is only recorded when it differs from the example value.
model_fields=()
collect_model_values() { # $1 = defaults source ("" = none), $2 = "save"|"one-off"
  local seed=$1 mode=$2
  local mh="" ch="" si="" ak="" ap="" smn=""
  if [[ -n $seed ]]; then
    while IFS=$'\t' read -r k v; do
      case $k in
        MODEL_HOST_PATH)  mh=$v ;;
        CACHE_HOST_PATH)  ch=$v ;;
        SERVING_IMAGE)    si=$v ;;
        API_KEY)          ak=$v ;;
        API_PORT)         ap=$v ;;
        SERVED_MODEL_NAME) smn=$v ;;
      esac
    done < <(read_kv_pairs "$seed")
  fi
  [[ -n $ap ]] || ap=$(example_value API_PORT)
  [[ -n $smn ]] || smn=$(example_value SERVED_MODEL_NAME)
  local desc="one-off (not saved)"
  [[ $mode == save ]] && desc="profiles/$stem-profile.env"
  printf 'model values -> %s\n' "$desc"
  ask_path "MODEL_HOST_PATH (abs path to model snapshot)" "$mh"
  model_fields+=("MODEL_HOST_PATH"$'\t'"$ans")
  ask_path "CACHE_HOST_PATH (abs path to writable cache)" "$ch"
  model_fields+=("CACHE_HOST_PATH"$'\t'"$ans")
  ask "SERVING_IMAGE (tag or digest)" "$si"
  model_fields+=("SERVING_IMAGE"$'\t'"$ans")
  ask_opt "API_KEY (empty = no API key)" "$ak"
  model_fields+=("API_KEY"$'\t'"$ans")
  while :; do
    ask "API_PORT (HTTP port on rank 0)" "$ap"
    if [[ $ans =~ ^[0-9]+$ ]] && (( 10#$ans >= 1 && 10#$ans <= 65535 )); then break; fi
    printf '  enter a number between 1 and 65535\n' >&2
  done
  model_fields+=("API_PORT"$'\t'"$ans")
  if [[ -n $smn ]]; then
    ask_yn "Change SERVED_MODEL_NAME '$smn'?"
    if [[ $yn == y ]]; then
      ask "New SERVED_MODEL_NAME"
      model_fields+=("SERVED_MODEL_NAME"$'\t'"$ans")
    elif [[ $smn != "$(example_value SERVED_MODEL_NAME)" ]]; then
      model_fields+=("SERVED_MODEL_NAME"$'\t'"$smn")
    fi
  else
    ask "SERVED_MODEL_NAME"
    model_fields+=("SERVED_MODEL_NAME"$'\t'"$ans")
  fi
}

if [[ -f $profile ]]; then
  declare -A prof=()
  while IFS=$'\t' read -r k v; do prof[$k]=$v; done < <(read_kv_pairs "$profile")
  missing=()
  for k in MODEL_HOST_PATH CACHE_HOST_PATH SERVING_IMAGE; do
    [[ -n ${prof[$k]:-} ]] || missing+=("$k")
  done
  if (( ${#missing[@]} > 0 )); then
    printf 'env-create: warning: %s is missing: %s — prompting, then adding them\n' \
      "$profile" "${missing[*]}" >&2
  fi
  printf 'using profile: %s\n' "$profile"
  collect_model_values "$profile" save
  # persist the answered values (SERVED_MODEL_NAME only when changed) into the profile
  { for pair in "${model_fields[@]}"; do printf '%s\n' "$pair"; done; } \
    | apply_kv append "$profile"
  printf 'updated: %s\n' "$profile"
else
  printf 'no profile: %s\n' "$profile"
  ask_yn "Create it?"
  if [[ $yn == y ]]; then
    # seeding offer: a filled rank-#.env in the model dir?
    seed_file=
    shopt -s nullglob
    candidates=("$model_dir"/rank-0.env "$model_dir"/rank-1.env)
    shopt -u nullglob
    for f in "${candidates[@]}"; do
      [[ -f $f ]] || continue
      ok=1
      while IFS=$'\t' read -r k v; do
        case $k in
          MODEL_HOST_PATH|CACHE_HOST_PATH|SERVING_IMAGE) [[ -n $v ]] || ok= ;;
        esac
      done < <(read_kv_pairs "$f")
      if [[ $ok ]]; then seed_file=$f; break; fi
    done
    seed_mode=
    if [[ -n $seed_file ]]; then
      ask_yn "Seed from existing ${seed_file##*/}?"
      [[ $yn == y ]] && seed_mode=$seed_file
    fi
    collect_model_values "$seed_mode" save
    {
      printf '# %s — model values consumed by env-create.sh (env stem: %s).\n' "$stem-profile.env" "$stem"
      for pair in "${model_fields[@]}"; do
        printf '%s=%s\n' "${pair%%$'\t'*}" "${pair#*$'\t'}"
      done
    } > "$profile"
    printf 'wrote: %s\n' "$profile"
  else
    collect_model_values "" one-off
  fi
fi

# --- 7. assemble -------------------------------------------------------------
target=$model_dir/rank-$rank.env
tmp=$(mktemp) || die "mktemp failed"
cp "$env_example" "$tmp"

case $rank in
  0) vllm_host_ip=$master_addr ;;
  1) vllm_host_ip=$rank1_ip ;;
esac

{
  printf 'NODE_RANK\t%s\n' "$rank"
  printf 'MASTER_ADDR\t%s\n' "$master_addr"
  printf 'VLLM_HOST_IP\t%s\n' "$vllm_host_ip"
  for pair in "${answers[@]}"; do printf '%s\n' "$pair"; done
  for pair in "${model_fields[@]}"; do printf '%s\n' "$pair"; done
} | apply_kv skip "$tmp"

if grep -En '^[A-Z_0-9]+=.*<[A-Za-z0-9_]+>' "$tmp"; then
  printf 'env-create: unresolved placeholders remain in %s (see lines above)\n' "$target" >&2
  rm -f "$tmp"
  exit 20
fi

mv "$tmp" "$target"
printf 'wrote: %s\n' "$target"

# --- 8. post-check -----------------------------------------------------------
if [[ -n $pair-serve ]]; then
  printf 'checking: %s --check %s\n' "${pair-serve##*/}" "${target##*/}"
  if ! bash "$pair-serve" --check "$target"; then
    printf 'env-create: --check failed for %s\n' "$target" >&2
    exit 1
  fi
  if [[ $rank == 1 ]]; then
    printf 'next (start rank 1 first): ./%s --run %s\n' "${pair-serve##*/}" "${target##*/}"
    printf 'then on rank 0:            ./%s --run rank-0.env\n' "${pair-serve##*/}"
  else
    printf 'next (after rank 1):       ./%s --run %s\n' "${pair-serve##*/}" "${target##*/}"
  fi
else
  exit 0
fi
