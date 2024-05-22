export CUDA_VISIBLE_DEVICES=0,1
accelerate launch --multi_gpu --num_processes 2 --main_process_port 9901 train_bash.py \
    --stage sft \
    --model_name_or_path /model/Llama-2-7B-hf \
    --do_train \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir /model/strdialogue \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1500 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --bf16  \
    --quantization_bit 4  \
    --report_to wandb 