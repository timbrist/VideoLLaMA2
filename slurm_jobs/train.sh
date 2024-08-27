#!/bin/bash
#SBATCH --job-name=ragdriver
#SBATCH --account=project_2010633
#SBATCH --partition=gpusmall
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
## if local fast disk on a node is also needed, replace above line with:
#SBATCH --gres=gpu:a100:1,nvme:900
#
# export PATH="/projappl/project_2010633/Video-LLaVA/videollava_evn/bin:$PATH"
export WORKSPACE=$(pwd)

export CACHESPACE=/scratch/project_2010633/videollama2

export PATH=${WORKSPACE}/vlm2_env/bin:$PATH
export HF_DATASETS_CACHE=${CACHESPACE}
export XDG_CACHE_HOME=${CACHESPACE}
export PIP_CACHE_DIR=${CACHESPACE}
export HF_HOME=${CACHESPACE}


bash ${WORKSPACE}/scripts/vllava/finetune.sh