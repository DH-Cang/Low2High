from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
from PIL import Image

class CDH_ObjaversePbrDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.SUBSET_NUM = 6
        self.VIEWS_PER_MESH = 16

        # check whether dataset file is valid
        directory_list = [
            "blip_prompts",
            "training_0_512",
            "training_1_512",
            "training_2_512",
            "training_3_512",
            "training_4_512",
            "training_5_512",
        ]
        for i, directory in enumerate(directory_list):
            full_path = os.path.join(root_dir, directory)
            assert os.path.isdir(full_path)

        # calculate dataset length
        self.subset_length = []
        for i in range(self.SUBSET_NUM):
            file_path = os.path.join(root_dir, f"training_{i}_512", f"data")
            sub_dirs = sorted(os.listdir(file_path))
            self.subset_length.append(len(sub_dirs))


    def __len__(self):
        length = 0
        for i in range(self.SUBSET_NUM):
            length += self.subset_length[i]
        return length * self.VIEWS_PER_MESH


    def __getitem__(self, idx):
        # mod idx by 16 to determine which view
        view_idx = idx % self.VIEWS_PER_MESH
        idx = idx // self.VIEWS_PER_MESH

        # calculate subset and subset offset
        accumulate_subset_length = 0
        for i in range(self.SUBSET_NUM):
            accumulate_subset_length += self.subset_length[i]
            if idx < accumulate_subset_length:
                subset_id = i
                subset_id_offset = idx - (accumulate_subset_length - self.subset_length[i])
                break
        assert subset_id is not None

        path = os.path.join(self.root_dir, f"training_{subset_id}_512", f"data")
        subset_dirs = sorted(os.listdir(path))
        uid = subset_dirs[subset_id_offset]

        prompt_file_path = os.path.join(self.root_dir, f"blip_prompts", f"training_{subset_id}.json")
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            text = data[uid]

        img_path = os.path.join(self.root_dir, f"training_{subset_id}_512", "data", uid, "normal", f"{view_idx:03d}.png")
        image = Image.open(img_path)

        sample = {'image': image, 'text': text}

        if self.transform:
            sample = self.transform(sample)

        return sample
    


# test code
import random
from datasets import load_dataset
if __name__ == "__main__":
    my_dataset = CDH_ObjaversePbrDataset("/home/panxiaoyu/disk/texture/dataset")
    print(len(my_dataset))

    random_integers = [random.randrange(0, len(my_dataset)) for _ in range(1000)]
    for _, index in enumerate(random_integers):
        print(my_dataset[index])
    # for i in range(32):
    #     sample = my_dataset[i]
    #     print(sample)
    #     sample["image"].save(f"./testoutput/{i}.png")
        
    # dataset = CDH_ObjaversePbrDataset("/home/panxiaoyu/disk/texture/dataset")
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # for step, batch in enumerate(dataloader):
    #     print(step)
    #     print(batch)
    # print("end")


