#  python distributed.py \
# 	--num_gpus 7 --gpu_list 1 2 3 4 5 6 7  --mode ortho_gt    \
# 	--workers_per_gpu 10 \
# 	--ortho_scale 1.35 \
# 	--input_models_path ../uid_list/pbr_uid_render.json  \
# 	--objaverse_root /data/zyq/data/pbr_glb \
# 	--save_folder /data/zyq/data/pbr_out_rendering \
# 	--resolution 1024 \
# 	--blender_install_path /data/zyq/blender_install

#  python distributed.py \
# 	--num_gpus 7 --gpu_list 1 2 3 4 5 6 7 --mode ortho_pos    \
# 	--workers_per_gpu 10 \
# 	--ortho_scale 1.35 \
# 	--input_models_path ../uid_list/pbr_uid_render.json \
# 	--objaverse_root /data/zyq/data/pbr_glb \
# 	--save_folder /data/zyq/data/pbr_out_rendering \
#     --resolution 1024 \
# 	--blender_install_path /data/zyq/blender_install

#  python distributed.py \
# 	--num_gpus 7 --gpu_list 1 2 3 4 5 6 7 --mode ortho_random    \
# 	--workers_per_gpu 10 \
# 	--ortho_scale 1.35 \
# 	--input_models_path ../uid_list/pbr_uid_render.json  \
# 	--objaverse_root /data/zyq/data/pbr_glb \
# 	--save_folder /data/zyq/data/pbr_out_rendering \
#   	--resolution 1024 \
# 	--blender_install_path /data/zyq/blender_install

 python distributed.py \
	--num_gpus 6 --gpu_list 2 3 4 5 6 7 --mode ortho_pos    \
	--workers_per_gpu 10 \
	--ortho_scale 1.35 \
	--input_models_path /data/zyq/CDH_training_Low2High/error_uid.json  \
	--objaverse_root /data/zyq/data/lvis_glb \
	--save_folder /data/zyq/data/lvis_out_rendering/normal_image \
  	--resolution 512 \
	--blender_install_path /data/zyq/blender_install_v2