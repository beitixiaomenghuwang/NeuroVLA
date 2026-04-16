#!/bin/bash
# Training script for NeuroVLA on TeleAvatar pick-marker task
# Run from project root: bash scripts/run_scripts/run_teleavatar_train.sh

# ── User settings ──────────────────────────────────────────────────────────────
# Local path to Qwen3-VL-4B-Instruct weights
MODEL_PATH=/home/caslx/Model/Qwen3-VL-4B-Instruct

# Absolute path to the dataset directory (pick_marker_put_into_cup_20251113)
DATA_ROOT_DIR=/home/caslx/Robotics/NeuroVLA/pick_marker_put_into_cup_20251113

RUN_ROOT_DIR=./results/Checkpoints
RUN_ID=teleavatar_pick_marker_$(date +%m%d_%H%M)

# GPU configuration (adjust CUDA_VISIBLE_DEVICES and --num_processes to your setup)
export CUDA_VISIBLE_DEVICES=6,7
NUM_GPUS=2

# W&B settings (optional – set WANDB_MODE=disabled to skip)
# export WANDB_MODE=disabled
WANDB_PROJECT=neurovla_teleavatar
WANDB_ENTITY=your_wandb_entity   # ← replace or set via env var

# ── Advanced overrides (can also be edited in the YAML) ────────────────────────
PER_DEVICE_BS=8
MAX_TRAIN_STEPS=30000
SAVE_INTERVAL=5000
GRADIENT_ACC=1

# ── Setup ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR=${RUN_ROOT_DIR}/${RUN_ID}
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"   # archive this script with the run

echo "============================================"
echo "  Run ID: ${RUN_ID}"
echo "  Model:  ${MODEL_PATH}"
echo "  Data:   ${DATA_ROOT_DIR}"
echo "  GPUs:   ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} processes)"
echo "============================================"

# ── Launch ─────────────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH}"

accelerate launch \
  --num_processes ${NUM_GPUS} \
  --main_process_port 29501 \
  --mixed_precision bf16 \
  NeuroVLA/training/train_NeuroVLA.py \
  --config_yaml NeuroVLA/config/training/neurovla_teleavatar.yaml \
  --framework.qwenvl.base_vlm "${MODEL_PATH}" \
  --datasets.vla_data.data_root_dir "$(dirname "${DATA_ROOT_DIR}")" \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BS} \
  --trainer.max_train_steps ${MAX_TRAIN_STEPS} \
  --trainer.save_interval ${SAVE_INTERVAL} \
  --trainer.gradient_accumulation_steps ${GRADIENT_ACC} \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}"
