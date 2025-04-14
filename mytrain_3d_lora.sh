#!/bin/bash
#SBATCH --account=rrg-mgoubran
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=neurosam2d-single-gpu%j.log
#SBATCH --job-name=neurosam2d-single-gpu

ulimit -n 51200

module load python/3.12
module load scipy-stack opencv cuda/12.2
source ~/NeuroSAM2D/bin/activate

which python
pwd

export WANDB_MODE="offline"

NIFTI_DATA_PATH="/home/lifeifei/projects/rrg-mgoubran/NeuroSAM_data/data"

IMAGE_SIZE=512
SAM_CONFIG="sam2_hiera_t_lora_512"
VIDEO_LENGTH=4
BATCH_SIZE=2
SEED=2025
MAX_TARGETS=3
EXP_NAME="NeuroSAM2D-Single-GPU-LORA"

python -u train_3d.py \
    -exp_name "${EXP_NAME}-${IMAGE_SIZE}-${VIDEO_LENGTH}-${BATCH_SIZE}-${SEED}" \
    -vis True \
    -pretrain "checkpoints/MedSAM2_pretrain.pth" \
    -val_freq 1 \
    -gpu True \
    -gpu_device 0 \
    -image_size $IMAGE_SIZE \
    -out_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    -num_workers 8 \
    -dataset "neurosam" \
    -sam_ckpt "checkpoints/MedSAM2_pretrain.pth" \
    -video_length $VIDEO_LENGTH \
    -data_path $NIFTI_DATA_PATH \
    -sam_config $SAM_CONFIG \
    -seed $SEED \
    -max_targets $MAX_TARGETS \
    --finetune_backbone \
    --finetune_neck \
    -lr_vit 5e-5