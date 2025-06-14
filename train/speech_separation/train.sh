#!/bin/sh

gpu_id=0,1				# visible GPUs
n_gpu=2		# number of GPU used for training
network=MossFormer2_SS_16K
checkpoint_dir=checkpoints/$network						# leave empty if it's a new training, otherwise provide the 'log_name'
config_pth=config/train/${network}.yaml		# the config file, only used if it's a new training
train_from_last_checkpoint=0 #resume training from last checkpoint, 1 for true, 0 for false.
init_checkpoint_path=../../clearvoice/checkpoints/${network}/last_best_checkpoint.pt  #we support initialize a model from previous checkpoint or pretrained model. Provide the model path here (eg. ../../clearvoice/checkpoints/${network}/last_best_checkpoint.pt), otherwise set to None.

print_freq=10  # No. steps waited for printing info
checkpoint_save_freq=5000  #No. steps waited for saving new checkpoint

if [ ! -d "${checkpoint_dir}" ]; then
  mkdir -p ${checkpoint_dir}
fi

cp $config_pth $checkpoint_dir/config.yaml

export PYTHONWARNINGS="ignore"
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES="$gpu_id"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1

python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=$n_gpu \
--master_port=$(date '+88%S') \
train.py \
--config ${config_pth} \
--checkpoint_dir ${checkpoint_dir} \
--train_from_last_checkpoint ${train_from_last_checkpoint} \
--init_checkpoint_path ${init_checkpoint_path} \
--print_freq ${print_freq} \
--checkpoint_save_freq ${checkpoint_save_freq}