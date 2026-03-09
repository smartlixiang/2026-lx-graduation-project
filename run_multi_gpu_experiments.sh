#!/usr/bin/env bash
set -u

PROJECT_DIR="$(pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
HOME_LOG="${HOME}/log.txt"

write_log() {
  local gpu="$1"
  local file="$2"
  local status="$3"
  printf "%s+%s+%s\n" "$gpu" "$file" "$status" >> "$HOME_LOG"
}

run_and_log() {
  local gpu="$1"
  local file="$2"
  shift 2
  if "$@"; then
    write_log "$gpu" "$file" "succeed"
    return 0
  else
    write_log "$gpu" "$file" "fail"
    return 1
  fi
}

prepare_env() {
  cd "$PROJECT_DIR" || return 1
  if [[ "${CONDA_DEFAULT_ENV:-}" == "shampoo" ]]; then
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
  else
    echo "[WARN] conda initialization not found" >&2
    return 1
  fi

  conda activate shampoo || return 1
}

run_seed_pipeline() {
  local gpu="$1"
  local seed="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"

  prepare_env || {
    write_log "$gpu" "prepare_env" "fail"
    return 1
  }

  run_and_log "$gpu" "calculate_my_mask.py" \
    python calculate_my_mask.py \
      --dataset cifar100 \
      --method group \
      --weight-group learned \
      --seeds "$seed"

  run_and_log "$gpu" "train_after_selection.py" \
    python train_after_selection.py \
      --dataset cifar100 \
      --mode learned_group \
      --seed "$seed"
}

run_gpu0_baseline() {
  local gpu="0"
  export CUDA_VISIBLE_DEVICES="$gpu"

  prepare_env || {
    write_log "$gpu" "prepare_env" "fail"
    return 1
  }

  run_and_log "$gpu" "x.py(naive)" \
    python x.py \
      --dataset cifar100 \
      --weight-group naive

  run_and_log "$gpu" "x.py(learned)" \
    python x.py \
      --dataset cifar100 \
      --weight-group learned
}

start_tmux_job() {
  local session_name="$1"
  local run_func="$2"
  shift 2

  if tmux has-session -t "$session_name" >/dev/null 2>&1; then
    echo "[WARN] tmux session '$session_name' already exists; skip starting duplicate job." >&2
    return 0
  fi

  tmux new-session -d -s "$session_name" \
    "cd '$PROJECT_DIR' && source '$SCRIPT_PATH' && $run_func $*"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found, please install tmux first." >&2
    exit 1
  fi

  # 5/6/7: 三个随机种子各占一张卡
  start_tmux_job 555 run_seed_pipeline 5 22 || exit 1
  start_tmux_job 666 run_seed_pipeline 6 42 || exit 1
  start_tmux_job 777 run_seed_pipeline 7 96 || exit 1

  # 0: CIFAR100 下 x.py 的 naive / learned
  start_tmux_job 000 run_gpu0_baseline || exit 1

  echo "Submitted tmux jobs: 000, 555, 666, 777"
  echo "Use 'tmux ls' to check sessions, and 'tmux attach -t <session>' to inspect progress in terminal."
fi
