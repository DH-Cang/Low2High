from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
from PIL import Image

class CDH_ObjaversePbrDataset(Dataset):


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
            "blip_prompt",
            "normal_image",
        ]
        for i, directory in enumerate(directory_list):
            full_path = os.path.join(root_dir, directory)
            assert os.path.isdir(full_path)

        # calculate dataset length
        file_path = os.path.join(root_dir, f"normal_image")
        uid_list = sorted(os.listdir(file_path))
        self.UID_NUM = len(uid_list)


    def __len__(self):
        return self.UID_NUM * self.VIEWS_PER_MESH


    def __getitem__(self, idx):
        # mod idx by 16 to determine which view
        view_idx = idx % self.VIEWS_PER_MESH
        idx = idx // self.VIEWS_PER_MESH

        file_path = os.path.join(self.root_dir, f"normal_image")
        uid_list = sorted(os.listdir(file_path))
        uid = uid_list[idx]

        prompt_file_path = os.path.join(self.root_dir, f"blip_prompt", f"merged_prompt.json")
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            assert uid in data
            text = data[uid]

        img_path = os.path.join(self.root_dir, f"normal_image", uid, f"normal_{view_idx}.webp")
        image = Image.open(img_path)

        # multiply alpha channel
        image_array = np.array(image)
        r, g, b, a = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2], image_array[:, :, 3]
        a[a > 0.0] = 1.0
        r = (r * a ).astype(np.uint8)
        g = (g * a ).astype(np.uint8)
        b = (b * a ).astype(np.uint8)
        new_image_array = np.stack([r, g, b], axis=-1)
        new_image = Image.fromarray(new_image_array, mode='RGB')

        sample = {'image': new_image, 'text': text, 'uid': uid}

        if self.transform:
            sample = self.transform(sample)

        return sample
    


# test code
import random
from datasets import load_dataset
if __name__ == "__main__":
    my_dataset = CDH_ObjaversePbrDataset("./normal_prompt_dataset")
    print(len(my_dataset))
    for i in range(len(my_dataset)):
        my_dataset[i].save("test.png")
    # random_integers = [random.randrange(0, len(my_dataset)) for _ in range(1000)]
    # for _, index in enumerate(random_integers):
    #     print(my_dataset[index])

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


