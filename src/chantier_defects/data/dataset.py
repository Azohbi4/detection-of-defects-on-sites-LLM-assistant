"""Dataset PyTorch et augmentations simulant les conditions de chantier."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToImage(),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            # Éclairage variable (soleil, ombre portée, lampe frontale)
            T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2),
            # Léger flou de bougé / mise au point
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transforms(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToImage(),
            T.Resize((img_size, img_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class DefectDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], transform, root: Path | None = None):
        self.paths = [Path(p) if root is None else root / p for p in df["path"]]
        # copie modifiable : avec le copy-on-write de pandas 3, to_numpy peut renvoyer une vue en lecture seule
        self.targets = df[classes].to_numpy(dtype=np.float32, copy=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as img:
            x = self.transform(img.convert("RGB"))
        return x, torch.from_numpy(self.targets[i])
