#!/usr/bin/env bash
set -euo pipefail

# Simple batch submitter for selected experiment settings.
#
# 1) Edit PARTITION and JOBS below.
# 2) Use --dry-run to preview commands.
# 3) Remove --dry-run to really submit.

WAIT_MODE=0
DRY_RUN=0
CONFIG_MODULE="configs.exp_config"
PARTITION="intel,amd"

usage() {
  cat <<'EOF'
Usage: ./submit_slurm_experiments.sh [options]

Options:
  --wait                 Submit each job with sbatch --wait (serial)
  --dry-run              Print commands only, do not submit
  --partition <name(s)>  One partition or comma list, e.g. intel or intel,amd
  --config <module>      Base config module passed to experiments.py --config
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait)
      WAIT_MODE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --config requires a value" >&2
        exit 1
      fi
      CONFIG_MODULE="$2"
      shift 2
      ;;
    --partition)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --partition requires a value" >&2
        exit 1
      fi
      PARTITION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$PROJECT_DIR/slurm_run_experiments.sbatch"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT_REL="results/experiments_${RUN_TS}"
RESULTS_ROOT_ABS="$PROJECT_DIR/$RESULTS_ROOT_REL"

if [[ -e "$RESULTS_ROOT_ABS" ]]; then
  RESULTS_ROOT_REL="results/experiments_${RUN_TS}_$$_$RANDOM"
  RESULTS_ROOT_ABS="$PROJECT_DIR/$RESULTS_ROOT_REL"
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH" >&2
  exit 1
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "ERROR: cannot find $SBATCH_SCRIPT" >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT_ABS"

choose_partition() {
  local requested="$1"

  # Single partition: use directly.
  if [[ "$requested" != *,* ]]; then
    echo "$requested"
    return
  fi

  # Comma-separated candidates: choose the one with most idle CPUs.
  local best_partition=""
  local best_idle=-1
  local raw
  local part
  local idle

  IFS=',' read -r -a parts <<< "$requested"
  for raw in "${parts[@]}"; do
    part="${raw//[[:space:]]/}"
    [[ -z "$part" ]] && continue

    idle="$(sinfo -h -p "$part" -o "%C" 2>/dev/null | awk -F'/' '{sum += $2} END {print sum + 0}')"
    if [[ -z "$idle" ]]; then
      idle=0
    fi

    if (( idle > best_idle )); then
      best_idle=$idle
      best_partition="$part"
    fi
  done

  if [[ -z "$best_partition" ]]; then
    best_partition="${parts[0]//[[:space:]]/}"
  fi

  echo "$best_partition"
}

# Format per line: setting_module|nodelist
# Leave nodelist empty to let Slurm pick a node in PARTITION.
JOBS=(
  # "configs.experiment_settings.barabasi_uniform_sweep_pi|"
  "configs.experiment_settings.barabasi_uniform_sweep_disturbance|"
  # "configs.experiment_settings.sbm_uniform_sweep_block_num|"
  # "configs.experiment_settings.watts_uniform_sweep_degree|"
)

echo "Project   : $PROJECT_DIR"
echo "Config    : $CONFIG_MODULE"
echo "Partition : $PARTITION"
echo "Root Dir  : $RESULTS_ROOT_REL"
echo "Jobs      : ${#JOBS[@]}"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Mode      : dry-run"
elif [[ $WAIT_MODE -eq 1 ]]; then
  echo "Mode      : wait (serial)"
else
  echo "Mode      : submit (non-blocking)"
fi

declare -i idx=0
for job in "${JOBS[@]}"; do
  idx+=1
  IFS='|' read -r setting nodelist <<< "$job"
  selected_partition="$(choose_partition "$PARTITION")"
  setting_subdir="${setting##*.}"
  setting_dir_rel="$RESULTS_ROOT_REL/$setting_subdir"
  setting_dir_abs="$PROJECT_DIR/$setting_dir_rel"
  out_file_rel="$setting_dir_rel/experiments_%j.out"
  err_file_rel="$setting_dir_rel/experiments_%j.err"

  if [[ $DRY_RUN -ne 1 ]]; then
    mkdir -p "$setting_dir_abs"
  fi

  cmd=(sbatch)
  if [[ $WAIT_MODE -eq 1 ]]; then
    cmd+=(--wait)
  fi
  cmd+=("--partition=$selected_partition")
  cmd+=("--output=$out_file_rel" "--error=$err_file_rel")
  cmd+=("--export=ALL,FW_RESULTS_ROOT=$RESULTS_ROOT_REL,FW_SETTING_SUBDIR=$setting_subdir")
  if [[ -n "$nodelist" ]]; then
    cmd+=("--nodelist=$nodelist")
  fi
  cmd+=(
    "$SBATCH_SCRIPT"
    "--setting" "$setting"
    "--config" "$CONFIG_MODULE"
  )

  printf "\n[%d/%d] [partition=%s] [setting=%s] " "$idx" "${#JOBS[@]}" "$selected_partition" "$setting_subdir"
  printf "%q " "${cmd[@]}"
  printf "\n"

  if [[ $DRY_RUN -eq 1 ]]; then
    continue
  fi

  submit_out="$(cd "$PROJECT_DIR" && "${cmd[@]}")"
  echo "$submit_out"
done

echo "All submissions processed."

#### Usage examples:
### preview commands without submitting:
# ./submit_slurm_experiments.sh --dry-run
### auto-select one partition from intel/amd for each submission:
# ./submit_slurm_experiments.sh
### submit only to amd partition:
# ./submit_slurm_experiments.sh --partition amd
### submit with serial waiting:
# ./submit_slurm_experiments.sh --wait
