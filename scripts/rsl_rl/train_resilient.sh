#!/usr/bin/env bash
# 训练看门狗：崩溃后自动从最新 checkpoint 续训。
#
# 为什么需要：rsl_rl 的 PPO 偶发以
#     RuntimeError: normal expects all elements of std >= 0.0
# 退出。本仓库已经撞过四次（velocitywithid_yaw @13900、perceptive_blind @12723、
# perceptive heightscan @3161、heightscan_logstd @3183）。崩溃前一迭代各项指标都正常
# （value loss ~0.014、noise std 0.43、reward 30），环境本身在 800 步 × 256 环境、
# 地形 0~9 级下没有任何非有限值，自适应学习率上限 1e-2 配合 max_grad_norm=1.0 单步
# 最多挪动 1e-2，不可能把有限参数推成 inf —— 所以是**某次更新出现 NaN 梯度**
# （PPO 的 ratio = exp(new_logp - old_logp) 溢出成 inf 是已知的可能来源），
# clip_grad_norm_ 对 NaN 梯度无能为力，参数整体变 NaN，下一次 act() 才报错。
#
# 根因没定位之前，用重启把损失限制在一个 save_interval 内（BasePPORunnerCfg 是 100 迭代）。
#
# 用法：
#   scripts/rsl_rl/train_resilient.sh <task> <experiment_name> <max_iterations> [其它 train.py 参数...]
# 例：
#   scripts/rsl_rl/train_resilient.sh Unitree-G1-29dof-PerceptiveHeightScan \
#       unitree_g1_29dof_perceptive 18100 --headless --run_name heightscan_logstd
#
# 首次运行若实验目录下已有 checkpoint，会直接从最新的那个续训；想从头训就换 experiment_name。
set -u

if [ $# -lt 3 ]; then
    sed -n '2,25p' "$0"
    exit 1
fi

TASK=$1
EXPERIMENT=$2
MAX_ITERATIONS=$3
shift 3

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_ROOT="$PROJ_DIR/logs/rsl_rl/$EXPERIMENT"
PYTHON="${UNITREE_RL_PYTHON:-/home/zxh/miniconda3/envs/ustc_isaaclab/bin/python}"
MAX_RESTARTS="${MAX_RESTARTS:-40}"

# 最新的 (run 目录名, checkpoint 文件名, 迭代号)；没有 checkpoint 时全部为空
latest_checkpoint() {
    [ -d "$LOG_ROOT" ] || return 0
    local newest
    newest=$(find "$LOG_ROOT" -mindepth 2 -maxdepth 2 -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
             | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$newest" ] || return 0
    local file run iter
    file=$(basename "$newest")
    run=$(basename "$(dirname "$newest")")
    iter=${file#model_}
    iter=${iter%.pt}
    echo "$run" "$file" "$iter"
}

for attempt in $(seq 0 "$MAX_RESTARTS"); do
    unset REMAINING
    read -r RUN CKPT ITER <<< "$(latest_checkpoint)"

    RESUME_ARGS=()
    if [ -n "${RUN:-}" ]; then
        if [ "$ITER" -ge "$((MAX_ITERATIONS - 1))" ]; then
            echo "[watchdog] 已到 $ITER / $MAX_ITERATIONS 迭代（$RUN/$CKPT），结束。"
            exit 0
        fi
        echo "[watchdog] 第 $attempt 次启动：从 $RUN/$CKPT（迭代 $ITER）续训到 $MAX_ITERATIONS"
        # train.py 的 --max_iterations 是"再跑多少迭代"，不是目标迭代号
        # （OnPolicyRunner.learn: tot_iter = current + num），续训时要减掉已完成的
        REMAINING=$((MAX_ITERATIONS - ITER))
        RESUME_ARGS=(--resume --load_run "$RUN" --checkpoint "$CKPT")
    else
        echo "[watchdog] 第 $attempt 次启动：$LOG_ROOT 下没有 checkpoint，从头训练"
    fi

    REMAINING=${REMAINING:-$MAX_ITERATIONS}
    STAMP=$(date +%Y-%m-%d_%H-%M-%S)
    RUN_LOG="$PROJ_DIR/logs/${EXPERIMENT}_watchdog_${STAMP}.log"
    echo "$RUN_LOG" > "$PROJ_DIR/logs/${EXPERIMENT}_current.txt"
    echo "[watchdog] 日志：$RUN_LOG"

    "$PYTHON" "$PROJ_DIR/scripts/rsl_rl/train.py" --task "$TASK" \
        --max_iterations "$REMAINING" "${RESUME_ARGS[@]}" "$@" > "$RUN_LOG" 2>&1
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "[watchdog] 训练正常结束。"
        exit 0
    fi

    if grep -q "normal expects all elements of std" "$RUN_LOG"; then
        REASON="std NaN（已知的 PPO NaN 梯度问题）"
    elif grep -q "out of memory\|CUDA error" "$RUN_LOG"; then
        # 显存问题重启多半也救不回来，交给人处理，避免空转
        echo "[watchdog] CUDA/显存错误，不自动重启。见 $RUN_LOG"
        exit $STATUS
    else
        REASON="退出码 $STATUS"
    fi
    echo "[watchdog] 训练中断：$REASON；最多丢 save_interval 个迭代，准备重启。"
    sleep 20  # 等上一进程把显存交还给 MPS，否则新进程会报 "MPS server is not ready"
done

echo "[watchdog] 重启次数达到上限 $MAX_RESTARTS，停止。"
exit 1
