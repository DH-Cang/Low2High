from torchvision import transforms
from CDH_objaverse_dataset_V3 import CDH_ObjaversePbrDataset
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import itertools

train_dataset = CDH_ObjaversePbrDataset("/data/cdh_dataset")

train_transforms = transforms.Compose(
        [
            # transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

tokenizer = CLIPTokenizer.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer", revision=None
)

def collate_fn(examples):
    # preprocess: image PIL to tensor; text prompt to tokenized id
    for example in examples:
        try:
            image = example['image']
            example["image_tensor"] = train_transforms(image)
            example["input_ids"] = tokenizer(
                example["text"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
        except Exception as e:
            print(f"error in uid {example['uid']}")

    image_tensor = torch.stack([example["image_tensor"] for example in examples])
    image_tensor = image_tensor.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"image_tensor": image_tensor, "input_ids": input_ids}


train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=32,
    num_workers=32,
)

print("===================== start =========================")
pbar = tqdm(total=len(train_dataloader), desc="Processing", unit="batch")
for batch_idx, batch_data in enumerate(train_dataloader):
    if batch_data["image_tensor"].shape != torch.Size([32, 3, 512, 512]) or batch_data["input_ids"].shape != torch.Size([32, 1, 77]):
        print(f"error in batch {batch_idx}")
    pbar.update(1)
pbar.close()


