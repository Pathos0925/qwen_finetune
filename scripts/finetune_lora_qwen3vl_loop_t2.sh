#!/bin/bash
# LoopLM Stage 1 finetune on Qwen3-VL-2B at T_max=2.
#
# Why qwen3_vl over qwen3_5: Qwen3-VL is pure full-attention decoder
# (RoPE + SwiGLU + pre-norm RMSNorm) — architecturally close to what
# the Ouro paper actually studied. Qwen3.5's hybrid SSM/attention
# layers compress sequences into recurrent state, which doesn't
# obviously benefit from re-application.
#
# Sized for a single A100 80GB. The 2B model leaves comfortable
# memory headroom — feel free to bump batch_per_device to 4 if
# nvidia-smi shows < 50 GB used.

MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
RUN_NAME="qwen3vl-2b-lora-loop-t2"

export PYTHONPATH=src:$PYTHONPATH
export WANDB_PROJECT="qwen-vl-finetune"
export WANDB_RUN_NAME="$RUN_NAME"

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=2
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

python src/train/train_sft.py \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --model_id $MODEL_NAME \
    --data_path train.json \
    --image_folder /tmp \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/qwen3vl_2b_lora_loop_t2 \
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
    --max_seq_length 4096 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --loop_enable True \
    --loop_t_max 2 \
    --loop_beta 0.1 \
    --loop_stage 1 \
    --loop_inter_norm True \
    --loop_kv_cache_strategy last \
    --loop_compile_layers False \
    --loop_compile_mode default
