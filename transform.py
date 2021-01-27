import random
import typing as T

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import kornia.augmentation as KA


class RandomRotation90(nn.Module):
    """Rotate by one of the given angles."""

    def forward(self, x: torch.Tensor):
        if random.random() < 0.5:
            x = x.transpose(-1, -2)
        return x
