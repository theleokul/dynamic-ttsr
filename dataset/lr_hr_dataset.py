import sys
import pathlib
import functools
import typing as T
import PIL
import numbers

import numpy as np
import cv2 as cv
import skimage.exposure as SKE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tv_transforms
import kornia
import kornia.augmentation as KA
import pytorch_lightning as pl

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
# import utils
import custom_transform


class LRHRDataset(data.Dataset):

    _extensions = ['png', 'jpg']
    _PIL2Tensor = tv_transforms.ToTensor()
    _compose_as = {
        'nn.Sequential': nn.Sequential
        , 'tv_transforms.Compose': tv_transforms.Compose
    }
    _transform_module = {
        'KA': KA,
        'tv_transforms': tv_transforms
    }

    @classmethod
    def _parse_transform(cls, transform: T.Dict) -> T.Optional[T.Callable]:
        """Parse transform description from a Dict and form a Callable object"""
        
        # Extract transform composer Callable
        compose_as = cls._compose_as[transform.pop('compose_as', 'nn.Sequential')]
        transforms = transform.get('transform', [])  # Extract the list of actual transform descriptions

        parsed_transforms = []
        for t in transforms:
            mod = cls._transform_module[t.pop('module', 'KA')]
            t = getattr(mod, t['transform'])(*t.get('args', []), **t.get('kwargs', {}))
            parsed_transforms.append(t)

        parsed_transforms = compose_as(*parsed_transforms) in len(parsed_transforms) > 0 else None

        return parsed_transforms

    @staticmethod
    def _make_sp_dims_even(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 != 0:
            x = x[..., :-1, :]
        if x.shape[-1] % 2 != 0:
            x = x[..., :-1]
        return x

    @classmethod
    def _parse_root(cls, root: T.Union[str, T.List[str]], repeat_count: int=1) -> T.List[str]:
        """Collect all image filepaths from the root/s"""

        paths = []
        if isinstance(root, str):
            for ext in cls._extensions:
                paths.extend([str(p) for p in pathlib.Path(root).rglob(f'*.{ext}')])
        else:
            for ext in cls._extensions:
                for r in root:
                    paths.extend([str(p) for p in pathlib.Path(r).rglob(f'*.{ext}')])

        paths = sorted(paths)
        paths *= repeat_count

        return paths

    @classmethod
    def _load_img(cls, path: str, to_tensor: bool=True) -> T.Union[torch.Tensor, PIL.Image.Image]:
        img = PIL.Image.open(path).convert('RGB')  # HACK: convert is need to uniformly handle grayscale images as RGB
        if to_tensor: 
            img = cls._PIL2Tensor(img)
        return img

    def __init__(
        self
        , input_root: T.Union[str, T.List[str]]
        , gt_root: T.Union[str, T.List[str]]
        , repeat_count: int=1
        , transform: T.Dict={}
        , to_tensor: bool=True
        , initial_input_scale_factor: T.Optional[float]=None
        , final_input_scale_factor: T.Optional[float]=None
        , make_sp_dims_even: bool=False
        , *args
        , **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.input_paths = self._parse_root(input_root, repeat_count)
        self.gt_paths = self._parse_root(gt_root, repeat_count)
        self.transform = self._parse_transform(transform)
        self.to_tensor = to_tensor
        self.initial_input_scale_factor = initial_input_scale_factor
        self.final_input_scale_factor = final_input_scale_factor
        self.make_sp_dims_even = make_sp_dims_even

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, index):
        x = self._load_img(self.input_paths[index], to_tensor=self.to_tensor)
        y = self._load_img(self.gt_paths[index], to_tensor=self.to_tensor)

        if self.make_sp_dims_even:
            x = self._make_sp_dims_even(x)
            y = self._make_sp_dims_even(y)

        if self.initial_input_scale_factor is not None:
            x = F.interpolate(x.unsqueeze(0), scale_factor=self.initial_input_scale_factor, mode='bilinear').squeeze(0)

        if self.transform is not None:
            x, y = self.transform(torch.stack([x, y]))

        if self.final_input_scale_factor is not None:
            x = F.interpolate(x.unsqueeze(0), scale_factor=self.final_input_scale_factor, mode='bilinear').squeeze(0)

        return x, y
