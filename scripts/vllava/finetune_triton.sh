#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

#echo "WORLD_SIZE: $WORLD_SIZE"
#echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=videollama2
RUN_NAME=vllava_settings
export DATA_DIR="/scratch/project_2010633/videollama2/video_process"
OUTP_DIR=work_dirs

export VIDEOLLAMA2_FOLDER="/home/yans2/videollama2_cache"
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export nproc_per_node=1
# vision_tower == vision encoder : https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0 torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $nproc_per_node \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama2/train_flash_attn.py \
    --deepspeed scripts/zero3.json \
    --model_type videollama2 \
    --model_path ${VIDEOLLAMA2_FOLDER}/Mistral-7B-Instruct-v0.2 \
    --vision_tower ${VIDEOLLAMA2_FOLDER}/clip-vit-large-patch14-336 \
    --mm_projector_type stc_connector \
    --pretrain_mm_mlp_adapter ${VIDEOLLAMA2_FOLDER}/VideoLLaMA2-7B-Base/mm_projector.bin \
    --data_path   ${VIDEOLLAMA2_FOLDER}/video_process/conv_base/conversation_bddx_train_processed.json \
    --data_folder ${VIDEOLLAMA2_FOLDER}/video_process/BDDX_Processed/ \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${VIDEOLLAMA2_FOLDER}/finetune_${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 99 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
