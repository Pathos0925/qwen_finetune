#!/bin/bash
# Stage 1 LoopLM finetune: joint LM (via LoRA) + exit-gate training.

MODEL_NAME="Qwen/Qwen3.5-4B"
RUN_NAME="qwen35-4b-lora-loop-stage1"

export PYTHONPATH=src:$PYTHONPATH
export WANDB_PROJECT="qwen-vl-finetune"
export WANDB_RUN_NAME="$RUN_NAME"

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

deepspeed --num_gpus=$NUM_DEVICES src/train/train_sft.py \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path train.json \
    --image_folder /tmp \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/qwen35_lora_loop_stage1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --run_name $RUN_NAME \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --loop_enable True \
    --loop_t_max 4 \
    --loop_beta 0.1 \
    --loop_stage 1 \
    --loop_inter_norm True \
    --loop_kv_cache_strategy last
