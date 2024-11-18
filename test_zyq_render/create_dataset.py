import os
import json
import subprocess

def find_glb_files(directory):
    glb_files = []
    uids = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.glb'):
                glb_files.append(os.path.join(root, file))
                uids.append(os.path.splitext(file)[0])
    return glb_files, uids

if __name__ == "__main__":
    glb_file_root_path = "/data/zyq/data/lvis_glb"
    glb_files, uids = find_glb_files(glb_file_root_path)
    prompt_file_path = "/data/zyq/CDH_training_Low2High/normal_prompt_dataset/blip_prompt/merged_prompt.json"
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        prompt_data = json.load(file)
    
    filtered_glb_file_path = []
    filtered_uid = []
    no_prompt_num = 0
    for i, uid in enumerate(uids):
        if uid not in prompt_data:
            no_prompt_num += 1
        else:
            filtered_glb_file_path.append(glb_files[i])
            filtered_uid.append(uids[i])
    
    output_file = "./valid_uid.json"
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(filtered_uid, file, ensure_ascii=False, indent=4)
    
    # env = os.environ.copy()
    # for i, glb_file_path in enumerate(filtered_glb_file_path):
    #     render_script = "/data/zyq/CDH_training_Low2High/test_zyq_render/run.sh"
    #     env['GLB_FILE_PATH'] = glb_file_path
    #     result = subprocess.run(['bash', render_script], env=env, capture_output=True, text=True)
    #     if result.returncode == 0:
    #         print(f"render glb file {i+1} / {len(filtered_glb_file_path)}")
    #     else:
    #         print(result.stderr)
    #         exit()

    # print("Dataset Creation Complete")

