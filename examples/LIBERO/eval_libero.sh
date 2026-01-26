#!/bin/bash

# ========== 环境配置 ==========
export LIBERO_HOME=/workspace/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # 让 eval_libero 找到 LIBERO 工具
export PYTHONPATH=$(pwd):${PYTHONPATH}        # 让 LIBERO 找到 websocket 工具

# ========== 实验参数 ==========
your_ckpt=/workspace/nature_submit/NeuroVLA/playground/Checkpoints/1104_neurovla_gru_xiaonao_goal_dualimage_spike_multistep_ac8_768*2_yibu/checkpoints/steps_30000_pytorch_model.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

task_suite_name=libero_goal
num_trials_per_task=20
video_out_path="results/${task_suite_name}/${folder_name}"

host="127.0.0.1"
base_port=10093
unnorm_key="franka"

# ========== 日志配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/eval_${folder_name}.log"

echo "============================================"
echo " LIBERO Evaluation Started "
echo " Time: ${TIMESTAMP}"
echo " Checkpoint: ${your_ckpt}"
echo " Log file: ${LOG_FILE}"
echo "============================================"

# ========== 执行主程序 ==========
# 保存标准输出和错误输出到日志文件，同时在终端打印
python ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.host "${host}" \
    --args.port "${base_port}" \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task "${num_trials_per_task}" \
    --args.video-out-path "${video_out_path}" \
    2>&1 | tee "${LOG_FILE}"

echo "============================================"
echo " 运行完成！日志保存在: ${LOG_FILE}"
echo "============================================"
