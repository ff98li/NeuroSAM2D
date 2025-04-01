#!/bin/bash
#SBATCH --account=rrg-mgoubran
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=neurosam2d-dist%j.log
#SBATCH --job-name=neurosam2d-dist

ulimit -n 51200

module load python/3.12
module load scipy-stack opencv cuda/12.2
source ~/NeuroSAM2D/bin/activate

which python
pwd

echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR

# Training setup
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE

## Master node setup
MAIN_HOST=`hostname`
export MASTER_ADDR=$MAIN_HOST

# Get a free port using python
export MASTER_PORT=$(python - <<EOF
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

export NNODES=$SLURM_NNODES
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES)) # M nodes x N GPUs
echo "nnodes: ${NNODES}"
echo "world_size: ${WORLD_SIZE}"

export TORCH_NCCL_ASYNC_HANDLING=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export WANDB_MODE="offline"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
echo SLURM_NTASKS=$SLURM_NTASKS

IMAGE_SIZE=384
SAM_CONFIG="sam2_hiera_t_384"
VIDEO_LENGTH=4
BATCH_SIZE=1
SEED=2025

NIFTI_DATA_PATH="/home/lifeifei/projects/rrg-mgoubran/NeuroSAM_data/data"
EXP_NAME="NeuroSAM2D-Dist-GPU"

srun torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_3d_dist.py \
    -exp_name "${EXP_NAME}-${IMAGE_SIZE}-${VIDEO_LENGTH}-${BATCH_SIZE}-${SEED}-Dist" \
    -vis True \
    -pretrain "checkpoints/MedSAM2_pretrain.pth" \
    -val_freq 1 \
    -gpu True \
    -image_size $IMAGE_SIZE \
    -out_size $IMAGE_SIZE \
    -dataset "neurosam" \
    -sam_ckpt "checkpoints/MedSAM2_pretrain.pth" \
    -video_length $VIDEO_LENGTH \
    -b $BATCH_SIZE \
    -data_path $NIFTI_DATA_PATH \
    -sam_config $SAM_CONFIG \
    -seed $SEED