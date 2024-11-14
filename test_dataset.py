# test code
import random
from datasets import load_dataset
from CDH_objaverse_dataset_V2 import CDH_ObjaversePbrDataset
if __name__ == "__main__":
    my_dataset = CDH_ObjaversePbrDataset("./normal_prompt_dataset")
    for i in range(len(my_dataset)):
        print(my_dataset[i])
        if i % 35000 == 0:
            my_dataset[i]["image"].save("./datasetcheck{i}.png")