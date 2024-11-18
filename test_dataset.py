import json
import os
from CDH_objaverse_dataset_V2 import CDH_ObjaversePbrDataset

def test_error_uid():
    root_dir = "/data/zyq/data/lvis_out_rendering/normal_image"

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
    my_dataset = CDH_ObjaversePbrDataset("/data/zyq/data/lvis_out_rendering")
    error_uid = my_dataset.check_completion()
    return error_uid


if __name__ == "__main__":
    error_uid_list = test_dataset()
    # test_error_uid()
    if len(error_uid_list) > 0:
        with open("./error_uid.json", 'w') as f:
            json.dump(error_uid_list, f)
    print(len(error_uid_list))
    

