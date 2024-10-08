#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem=32000M
#SBATCH --output=pi-gpu.out
#SBATCH --gres=gpu:a100:1

module load cuda/12.2.1
export WORKSPACE=$(pwd)
export CACHESPACE=$WRKDIR/videollama2_cache
export HF_DATASETS_CACHE=${CACHESPACE}
export XDG_CACHE_HOME=${CACHESPACE}
export PIP_CACHE_DIR=${CACHESPACE}
export HF_HOME=${CACHESPACE}



bash ${WORKSPACE}/scripts/vllava/finetune_triton.sh