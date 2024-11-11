export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export TRAIN_DATA_DIR="./normal_prompt_dataset"
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
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --use_ema \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="sd-normal-model"\
  --validation_prompts="a christmas tree made of paper"\
  --checkpointing_steps=10000\
  --num_train_epochs=1
