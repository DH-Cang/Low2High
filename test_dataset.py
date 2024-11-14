# test code
import random
from datasets import load_dataset
from CDH_objaverse_dataset_V2 import CDH_ObjaversePbrDataset
import json
if __name__ == "__main__":
    my_dataset = CDH_ObjaversePbrDataset("./normal_prompt_dataset")
    error_uid_list = []
    for i in range(len(my_dataset)):
        if my_dataset[i]["image"] is None and my_dataset[i]['uid'] not in error_uid_list:
            error_uid_list.append(my_dataset[i]['uid'])
        else:
            if i % 35000 == 0:
                my_dataset[i]["image"].save(f"./datasetcheck{i}.png")

    if len(error_uid_list) > 0:
        with open("./error_uid.json", 'w') as f:
            json.dump(error_uid_list, f)