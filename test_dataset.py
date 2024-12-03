import json
import os
from CDH_objaverse_dataset_V3 import CDH_ObjaversePbrDataset
import concurrent.futures
import time
from tqdm import tqdm
import threading

def test_error_uid():
    root_dir = "/data/cdh_dataset"

    with open("./error_uid.json", 'r') as file:
        data = json.load(file)

    error_list = []
    for i, uid in enumerate(data):
        dir = os.path.join(root_dir, uid)
        for i in range(16):
            path = os.path.join(dir, f"normal_{i}.webp")
            if os.path.exists(path) == False and uid not in error_list:
                error_list.append(uid)

    print(len(error_list))

def test_dataset():
    my_dataset = CDH_ObjaversePbrDataset("/data/cdh_dataset")
    error_uid = my_dataset.check_completion()
    return error_uid

def iterate():
    start_time = time.time()
    my_dataset = CDH_ObjaversePbrDataset("/data/cdh_dataset")
    with tqdm(total=len(my_dataset) // 16, desc="Processing", unit="batch") as pbar:
        for i in range(0, len(my_dataset), 16):
            img, text, uid = my_dataset[i]
            if img is None or text is None:
                print(f"{uid} error!")

            # 输出当前时间戳
            # current_time = time.time()
            # elapsed_time = current_time - start_time
            # print(f"Iteration {i}: Current timestamp: {current_time:.6f}, Elapsed time: {elapsed_time:.6f} seconds")

            pbar.update(1)



# pbar = tqdm(total=730720, desc="Processing", unit="item")
# # 创建锁
# lock = threading.Lock()
# # 模拟处理函数
# def process_item(index, dataset):
#     # 获取锁
#     lock.acquire()
#     try:
#         # 修改共享变量
#         pbar.update(1)
#     finally:
#         # 释放锁
#         lock.release()

    
#     item = dataset[index]
#     if item["image"] is None or item["text"] is None:
#         return item["uid"], False
#     else:
#         return item["uid"], True

# # 使用 ProcessPoolExecutor 处理数据集
# def process_dataset(dataset, num_processes=52):
#     results = []
    
#     with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
#         # 提交任务到进程池
#         futures = [executor.submit(process_item, index, dataset) for index in range(len(dataset))]
        
#         # 收集结果
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 uid, result = future.result()
#                 if result == False:
#                     print(uid)
#             except Exception as e:
#                 print(f"An error occurred: {e}")
    
#     return results


if __name__ == "__main__":
    # error_uid_list = test_dataset()
    # # test_error_uid()
    # if len(error_uid_list) > 0:
    #     with open("./error_uid.json", 'w') as f:
    #         json.dump(error_uid_list, f)
    # print(len(error_uid_list))

    iterate()

    # process_dataset(CDH_ObjaversePbrDataset("/data/cdh_dataset"))
    # pbar.close()


