from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
from PIL import Image
from tqdm import tqdm
from CDH_objaverse_dataset_V3 import CDH_ObjaversePbrDataset

my_dataset = CDH_ObjaversePbrDataset("/data/cdh_dataset")
# for i in range(423040, 425600):
#     item = my_dataset[i]
#     print(item)
image0 = Image.open("/data/cdh_dataset/94ee9b5b53894cd593462f18b2d71572/normal/000.png").convert('RGB')
print(image0.mode)

image = Image.open("/data/cdh_dataset/94ee9b5b53894cd593462f18b2d71572/normal/011.png").convert('RGB')
print(image.mode)
