from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
from PIL import Image
from tqdm import tqdm

class CDH_ObjaversePbrDataset(Dataset):


    # file struct
    # root_dir
    # ├── blip_prompt
    # │   └── merged_prompt.json
    # └── uid0
    #     └── normal
    #         ├── 000.png
    #         ├── 001.png
    #         ├── 002.png
    #         ...
    #         └── 015.png
    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.VIEWS_PER_MESH = 16

        # # check whether dataset file is valid
        # directory_list = [
        #     "blip_prompt",
        # ]
        # for i, directory in enumerate(directory_list):
        #     full_path = os.path.join(root_dir, directory)
        #     assert os.path.isdir(full_path)
        assert os.path.exists(os.path.join(root_dir, "correct_final_prompt.json"))

        # calculate dataset length
        uid_list = sorted(os.listdir(root_dir))
        self.UID_NUM = len(uid_list)


    def __len__(self):
        return self.UID_NUM * self.VIEWS_PER_MESH


    def __getitem__(self, idx):
        # mod idx by 16 to determine which view
        view_idx = idx % self.VIEWS_PER_MESH
        idx = idx // self.VIEWS_PER_MESH

        # file_path = os.path.join(self.root_dir, f"normal_image")
        uid_list = sorted(os.listdir(self.root_dir))
        uid = uid_list[idx]

        prompt_file_path = os.path.join(self.root_dir, "correct_final_prompt.json")
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            assert uid in data
            text = data[uid]

        img_path = os.path.join(self.root_dir, uid, f"normal_image", f"{view_idx:03d}.png")

        try:
            image = Image.open(img_path)

            # # multiply alpha channel
            # image_array = np.array(image)
            # r, g, b, a = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2], image_array[:, :, 3]
            # a[a > 0.0] = 1.0
            # r = (r * a ).astype(np.uint8)
            # g = (g * a ).astype(np.uint8)
            # b = (b * a ).astype(np.uint8)
            # new_image_array = np.stack([r, g, b], axis=-1)
            # new_image = Image.fromarray(new_image_array, mode='RGB')

            # sample = {'image': new_image, 'text': text, 'uid': uid}
            sample = {'image': image, 'text': text, 'uid': uid}
        except FileNotFoundError:
            sample = {'image': None, 'text': text, 'uid': uid}
            print(f"uid {uid} image not load correctly")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            exit()

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def check_completion(self):
        num = self.__len__()

        uid_list = sorted(os.listdir(self.root_dir))

        prompt_file_path = os.path.join(self.root_dir, "correct_final_prompt.json")
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            prompt_data = json.load(file)

        error_uid = []
        for idx in tqdm(range(num), desc="scanning data", total=num):
            # mod idx by 16 to determine which view
            view_idx = idx % self.VIEWS_PER_MESH
            idx = idx // self.VIEWS_PER_MESH
            uid = uid_list[idx]

            assert uid in prompt_data
            text = prompt_data[uid]

            img_path = os.path.join(self.root_dir, uid, f"normal_image", f"{view_idx:03d}.png")

            if os.path.exists(img_path) == False and uid not in error_uid:
                error_uid.append(uid)
        return error_uid


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


