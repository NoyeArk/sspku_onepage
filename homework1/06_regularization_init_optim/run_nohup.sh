#!/usr/bin/env bash
# 使用 nohup 在后台运行三个实验脚本，日志写入 logs/。
# 用法：
#   ./run_nohup.sh              # 依次启动三个任务（各自后台）
#   ./run_nohup.sh reg          # 仅正则化实验
#   ./run_nohup.sh init         # 仅初始化实验
#   ./run_nohup.sh optim        # 仅优化器实验

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$SCRIPT_DIR"

run_one() {
  local name="$1"
  local py="$2"
  local log="$LOG_DIR/${name}.log"
  # python3 重定向到文件时默认块缓冲，长时间无输出；-u 无缓冲，日志可实时 tail
  nohup python3 -u "$py" >>"$log" 2>&1 &
  echo "$!" >"$LOG_DIR/${name}.pid"
  echo "已启动 $py → PID $(cat "$LOG_DIR/${name}.pid")，日志: $log"
}

case "${1:-all}" in
  reg)
    run_one "train_regularization" "train_regularization.py"
    ;;
  init)
    run_one "train_initialization" "train_initialization.py"
    ;;
  optim)
    run_one "train_optimizers" "train_optimizers.py"
    ;;
  all|"")
    run_one "train_regularization" "train_regularization.py"
    run_one "train_initialization" "train_initialization.py"
    run_one "train_optimizers" "train_optimizers.py"
    ;;
  -h|--help)
    echo "用法: $0 [reg|init|optim|all]"
    echo "  默认 all：依次后台启动三个训练脚本。"
    exit 0
    ;;
  *)
    echo "未知参数: $1" >&2
    echo "用法: $0 [reg|init|optim|all]" >&2
    exit 1
    ;;
esac
