export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export TRAIN_DATA_DIR="/data/zyq/data/lvis_out_rendering"
export HF_ENDPOINT="https://hf-mirror.com"

# file struct
# root_dir
# ├── blip_prompt
# │   └── merged_prompt.json
# └── normal_image
#     └── uid0
#         ├── normal_0.webp
#         ├── normal_1.webp
#         ├── normal_2.webp
#         ...
#         └── normal_15.webp

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $TRAIN_DATA_DIR \
  --use_ema \
  --resolution 512 \
  --train_batch_size 100 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --mixed_precision "fp16" \
  --num_train_epochs 21 \
  --learning_rate 1e-05 \
  --max_grad_norm 1 \
  --lr_scheduler "constant" \
  --lr_warmup_steps 0 \
  --output_dir "sd-normal-model"\
  --validation_prompts "a christmas tree made of paper" "a bird sitting on a branch" "a pair of over-ear headphones"\
  --checkpointing_steps 2868 \
  --validation_epochs 3
  
  

  # python train_text_to_image.py \
  # --pretrained_model_name_or_path $MODEL_NAME \
  # --train_data_dir $TRAIN_DATA_DIR \
  # --use_ema \
  # --resolution 512 \
  # --train_batch_size 1 \
  # --gradient_accumulation_steps 4 \
  # --gradient_checkpointing \
  # --mixed_precision "fp16" \
  # --num_train_epochs 3 \
  # --learning_rate 1e-05 \
  # --max_grad_norm 1 \
  # --lr_scheduler "constant" \
  # --lr_warmup_steps 0 \
  # --output_dir "sd-normal-model"\
  # --validation_prompts "a christmas tree made of paper" "a bird sitting on a branch" "a pair of over-ear headphones"\
  # --checkpointing_steps 1\  # origin 10000 \
