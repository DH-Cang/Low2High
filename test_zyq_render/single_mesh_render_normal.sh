export GLB_FILE_PATH="/data/zyq/data/lvis_glb/fff87ccd24464db1adb0995b5415b2d4.glb"

blenderproc run --blender-install-path /data/zyq/blender_install_v2 \
                        /data/zyq/CDH_training_Low2High/test_zyq_render/ortho_pos_single.py \
                        --object_path $GLB_FILE_PATH \
                        --view 0 \
                        --output_folder /data/zyq/CDH_training_Low2High/normal_prompt_dataset/normal_image \
                        --ortho_scale 1.35 \
                        --resolution 512

# echo "$GLB_FILE_PATH"
