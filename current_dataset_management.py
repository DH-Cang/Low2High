import os
from pathlib import Path
import shutil
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
lock = Lock()

def get_folder_size(folder_path):
    """
    计算指定文件夹的总大小（以字节为单位）。

    :param folder_path: 文件夹的路径
    :return: 文件夹的总大小（字节）
    """
    total_size = 0
    for path in Path(folder_path).rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size

def format_size(size_bytes):
    """
    将字节大小格式化为易读的字符串（KB, MB, GB等）。

    :param size_bytes: 字节大小
    :return: 格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def calculate_normal_folders_size(data_folder):
    """
    计算 data 文件夹下所有 uid 子文件夹中的 normal 子文件夹的总大小。

    :param data_folder: data 文件夹的路径
    :return: 总大小（字节）
    """
    total_size = 0
    data_path = Path(data_folder)
    
    # 遍历 data 文件夹下的所有 uid 子文件夹
    for uid_path in data_path.iterdir():
        if uid_path.is_dir():
            normal_path = uid_path / 'normal'
            if normal_path.exists() and normal_path.is_dir():
                # 计算 normal 文件夹的大小
                size = get_folder_size(normal_path)
                total_size += size
    
    return total_size


def collect_normal_folders(src_dirs, dest_dir):
    """
    从多个源目录中收集所有 uid/normal 文件夹，并将它们复制到目标目录中。

    :param src_dirs: 源目录列表
    :param dest_dir: 目标目录
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for index, src_dir in enumerate(src_dirs):
        src_path = Path(src_dir)
        if not src_path.exists() or not src_path.is_dir():
            print(f"警告: 路径 '{src_dir}' 不存在或不是一个目录，已跳过。")
            continue

        subdirectories = [item for item in src_path.iterdir() if item.is_dir()]
        progress_bar = tqdm(total=len(subdirectories), desc=f"Copying Dataset {index}", unit="item")
        for uid_path in subdirectories:
            if uid_path.is_dir() and (uid_path / 'normal').exists():
                normal_path = uid_path / 'normal'
                dest_normal_path = Path(dest_dir) / uid_path.name / 'normal'
                shutil.copytree(normal_path, dest_normal_path, dirs_exist_ok=True)
                progress_bar.update(1)
                #print(f"复制文件夹 '{normal_path}' 到 '{dest_normal_path}'")
        progress_bar.close()

# def create_zip_archive(zip_filename, src_dir):
#     """
#     将指定目录中的所有文件和文件夹压缩成一个 ZIP 文件。

#     :param zip_filename: 压缩包的名称
#     :param src_dir: 要压缩的目录
#     """
#     total_files = sum(len(files) for _, _, files in os.walk(src_dir))
#     progress_bar = tqdm(total=total_files, desc=f"Compressing", unit="item")
#     with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         for root, _, files in os.walk(src_dir):
#             for file in files:
#                 file_full_path = os.path.join(root, file)
#                 arcname = os.path.relpath(file_full_path, src_dir)
#                 zipf.write(file_full_path, arcname=arcname)
#             progress_bar.update(1)
#     progress_bar.close()
#     print(f"生成 ZIP 文件 '{zip_filename}'")

def compress_file(zipf, file_full_path, arcname, progress_bar):
    """
    将单个文件压缩到 ZIP 文件中。

    :param zipf: ZIP 文件对象
    :param file_full_path: 文件的完整路径
    :param arcname: 文件在 ZIP 文件中的相对路径
    """
    with lock:
        zipf.write(file_full_path, arcname=arcname)
        progress_bar.update(1)

def create_zip_archive(zip_filename, src_dir, max_workers=32):
    """
    将指定目录中的所有文件和文件夹压缩成一个 ZIP 文件。

    :param zip_filename: 压缩包的名称
    :param src_dir: 要压缩的目录
    :param max_workers: 最大线程数
    """
    total_files = sum(len(files) for _, _, files in os.walk(src_dir))
    progress_bar = tqdm(total=total_files, desc=f"Compressing", unit="item")

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for root, _, files in os.walk(src_dir):
                for file in files:
                    file_full_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_full_path, src_dir)
                    future = executor.submit(compress_file, zipf, file_full_path, arcname, progress_bar)
                    futures.append(future)
            
            for future in futures:
                future.result()
                # progress_bar.update(1)
    
    progress_bar.close()
    print(f"生成 ZIP 文件 '{zip_filename}'")

# 检查normal是否都有合法的prompt，如果没有就补上
def check_prompts():
    import os
    import glob
    from tqdm import tqdm
    import concurrent.futures
    import subprocess
    import multiprocessing
    import requests
    from PIL import Image
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    import torch
    import json
    import warnings

    # processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", local_files_only=True)
    # model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, local_files_only=True)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)

    # load prompt json
    with open("/home/xuhao/DehanChen/normal_prompt_dataset/blip_prompt/merged_prompt.json", 'r', encoding='utf-8') as file:
        uid_prompt = json.load(file)

    # for each data dir
    count = 0
    uid_list = []
    for i in range(6):
        data_dir = f"/home/panxiaoyu/disk/texture/dataset/training_{i}_512/data"
        subdirectories = [item for item in Path(data_dir).iterdir() if item.is_dir()]
        # for each uid subdirectory
        for uid_path in subdirectories:
            uid = Path(uid_path).name
            uid_list.append(uid)
            if uid not in uid_prompt:
                print(f"Uid {uid} has no prompt")
                count += 1
                # add blip
    print(count)

    key_to_be_deleted = []
    for key in uid_prompt.keys():
        if key not in uid_list:
            print(f"Uid {key} has no image")
            key_to_be_deleted.append(key)

    for key in key_to_be_deleted:
        uid_prompt.pop(key)
    
    with open("/home/xuhao/DehanChen/normal_prompt_dataset/blip_prompt/correct_final_prompt.json", 'w', encoding='utf-8') as file:
        json.dump(uid_prompt, file, ensure_ascii=False, indent=4)


    # with warnings.catch_warnings():
    #     # 忽略所有警告
    #     warnings.simplefilter("ignore")
    #     image_path = os.path.join(image_save_dir, uid, "000.png")
    #     image = Image.open(image_path).convert('RGB').resize((512, 512))
    #     inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    #     generated_ids = model.generate(**inputs)
    #     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def main():
    # # 计算文件夹大小
    # total_normal_size = 0
    # for i in range(6):
    #     data_folder = f"/home/panxiaoyu/disk/texture/dataset/training_{i}_512/data"
    #     total_normal_size += calculate_normal_folders_size(data_folder)
    # formatted_total_size = format_size(total_normal_size)
    # print(f"所有 normal 文件夹的总大小为: {formatted_total_size}")


    # 打包成zip文件
    src_dirs = [
        '/home/panxiaoyu/disk/texture/dataset/training_0_512/data',
        '/home/panxiaoyu/disk/texture/dataset/training_1_512/data',
        '/home/panxiaoyu/disk/texture/dataset/training_2_512/data',
        '/home/panxiaoyu/disk/texture/dataset/training_3_512/data',
        '/home/panxiaoyu/disk/texture/dataset/training_4_512/data',
        '/home/panxiaoyu/disk/texture/dataset/training_5_512/data'
    ]
    dest_dir = '/home/xuhao/projects/xh-disk/temp_pack_dataset'
    zip_filename = '/home/xuhao/disk/cdh_data/cdh_collected_data.zip'

    # 收集所有 uid/normal 文件夹
    # collect_normal_folders(src_dirs, dest_dir)

    # 生成 ZIP 压缩包
    create_zip_archive(zip_filename, dest_dir)

    # 遍历所有normal图，检测是否有prompt，如果没有就补上
    # check_prompts()
    


if __name__ == "__main__":
    main()