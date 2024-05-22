CUDA_VISIBLE_DEVICES=0 python export_model.py \
    --model_name_or_path /model/Llama-2-7B-hf \
    --template llama2 \
    --finetuning_type lora \
    --checkpoint_dir /model/strdialogue \
    --export_dir /model/strdialogue-merge