#!/bin/bash

your_ckpt=/workspace/nature_submit/NeuroVLA/playground/Checkpoints/1104_neurovla_gru_xiaonao_goal_dualimage_spike_multistep_ac8_768*2_yibu/checkpoints/steps_30000_pytorch_model.pt

base_port=10093
# export DEBUG=true

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16