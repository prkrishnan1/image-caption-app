#!/bin/bash

#SBATCH --account=krishp20
#SBATCH --job-name=image-caption-model-train
#SBATCH --error=%x-%j.err
#SBATCH --exclusive
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --time=3-20:00:00

# setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# Additional setting for DGX Cloud
export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_TOPO_FILE=/cm/shared/etc/ndv4-topo.xml
export NCCL_DEBUG=INFO
export NCCL_PROTO=LL,LL128,Simple
export NCCL_ALGO=Tree,Ring,CollnetDirect,CollnetChain,NVLS
export MELLANOX_VISIBLE_DEVICES=all
export PMIX_MCA_gds=hash
export PMIX_MCA_psec=native

export SHARED_STORAGE_ROOT="/lustre/fs0/scratch/krishp20/"

export CONTAINER_IMAGE="/lustre/fs0/scratch/images/nvidia+pytorch+24.07-py3.sqsh"
export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

export OUTPUT_DIR=/output

# Language pair option
# Possible options are de-en,cs-en,is-en,zh-en,ru-en,en-de,en-cs,en-is,en-zh,en-ru
export PAIRS=zh-en

export LORA_RANK=16

srun -N 1 --container-image $CONTAINER_IMAGE --container-mounts $SHARED_STORAGE_ROOT:/workspace --container-workdir /workspace --no-container-mount-home --pty /bin/bash -c "source /workspace/bin/activate; echo 'Python Version: $(/workspace/bin/python3 --version)'; echo 'Installing required dependencies from requirements.txt'; /workspace/bin/python3 -m pip install -r /workspace/cocodatasetnew/cocodataset/requirements.txt; /workspace/bin/python3 /workspace/cocodatasetnew/cocodataset/train_image_captioning_model.py -w '/workspace/'"







