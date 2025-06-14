import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import ConvertImageDtype, Resize


class PizzaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [
            f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        self.converter = ConvertImageDtype(torch.float32)
        self.resizer = Resize((64, 64), antialias=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = read_image(img_path, mode=ImageReadMode.RGB)
        image = self.converter(image)
        image = self.resizer(image)

        if self.transform:
            image = self.transform(image)

        if "not_pizza" in img_name.lower():
            label = torch.tensor([0.0], dtype=torch.float32)
        elif "pizza" in img_name.lower():
            label = torch.tensor([1.0], dtype=torch.float32)
        else:
            raise ValueError(f"Не удалось определить класс для файла: {img_name}")

        return image, label


class NoisyPizzaDataset(Dataset):
    def __init__(self, base_dataset, noise_std=0.2):
        self.base = base_dataset
        self.noise_std = noise_std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _ = self.base[idx]
        noisy_image = image + torch.randn_like(image) * self.noise_std
        noisy_image = torch.clamp(noisy_image, 0.0, 255.0)
        return noisy_image, image


class SequencePizzaDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = img.permute(1, 2, 0).reshape(img.shape[1], -1)
        return img, label
