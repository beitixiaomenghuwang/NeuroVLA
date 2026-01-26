
# export HF_HOME=/mnt/petrelfs/share/yejinhui/Models/huggingface_cache

export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# # 用于check save 的时候的通信
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1000  # 超时时间设为 1 小时（单位：秒）
MODEL_PATH=/workspace/model/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 # must be a local path, due to simpler will run in other
data_root_dir=/workspace/dataset/libero_goal_no_noops_1.0.0_lerobot/datasets--IPEC-COMMUNITY--libero_goal_no_noops_1.0.0_lerobot/snapshots
run_root_dir=./playground/Checkpoints
run_id=1124_spikevla_spike_multisteps_MLPxiaonao_goal_dualimage_ac8_768
#export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

  # --pretrained_checkpoint ${MODEL_PATH} \

#export NCCL_SOCKET_IFNAME=bond0
#export NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=1,2,3,4
framework_name=spikevla_xiaonao
dataset_py=lerobot_datasets
data_mix=libero_goal
action_chunk=8
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29500 \
  NeuroVLA/training/train_NeuroVLA.py\
  --config_yaml /workspace/llavavla0/NeuroVLA/config/training/internvla_cotrain_custom.yaml \
  --framework.qwenvl.base_vlm ${MODEL_PATH} \
  --framework.name ${framework_name} \
  --framework.layer_qformer.num_query_tokens ${action_chunk} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.dataset_py ${dataset_py} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.max_train_steps 40000 \
  --trainer.save_interval 10000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project spikeVLA-MLP \
  --wandb_entity weiyuguo \
  # --is_debug True


