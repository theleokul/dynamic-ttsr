import sys
import pathlib
import functools
import typing as T
import PIL

import numpy as np
import cv2 as cv
import skimage.exposure as SKE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
import kornia
import pytorch_lightning as pl

DIR_PATH = pathlib.Path(__file__).resolve().parent()
sys.path.append(DIR_PATH)
import utils
import _pan as pan
import transform as custom_transform


class Image2ImageDataset(data.Dataset):

    _extensions = ['png', 'jpg']

    @staticmethod
    def _parse_root(root: str, repeat_count: int=1, sort: bool=True) -> T.List[str]:
        """
            Collect all image filepaths from root.
        """
        paths = []
        for ext in _extensions:
            paths.extend(Path(root).rglob(f'*.{ext}'))

        if sort:
            paths = sorted(paths)

        paths *= repeat_count

        return paths

    @staticmethod
    def _parse_transform(transform: T.Dict) -> T.Callable:
        """
            Parse transform description in Dict and form a Callable object.
        """

        # Parse metainfo
        compose_as = eval(transform.pop('compose_as', 'nn.Sequential'))
        transform = transform.get('transform', [])

        parsed_transform = []
        for t in transform:
            mod = eval(t.pop('module', 'KA'))
            t = getattr(mod, t['transform'])(*t.get('args', []), **t.get('kwargs', dict()))
            parsed_transform.append(t)
        parsed_transform = compose_as(parsed_transform)

        return parsed_transform

    @staticmethod
    def _load_img(path: str, dtype: torch.dtype=torch.float32, device=torch.device('cpu')) -> torch.Tensor:
        """
        Load an rgb image, scale it to [0., 1.] and spit out as torch.Tensor (CxHxW).
        """
    
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        img = SKE.rescale_intensity(img, out_range=(0., 1.))
        img = torch.as_tensor(img, dtype=dtype, device=device)
        img = img.permute(2, 0, 1)  # Move color channel to the first dimension

        # NOTE: make dimensions even (to eliminate shape problems after downsampling)
        h, w = img.shape[-2:]
        if h % 2 != 0:
            img = img[:, :-1]
        if w % 2 != 0:
            img = img[:, :, :-1]

        return img

    def __init__(
        self
        , gt_root: str
        , repeat_count: int=1
        , dtype: str='float32'
        , device: str='cpu'
        , scale_factor: float=0.5
        , interpolation_mode: str='bilinear'
        , transform: T.Dict=dict()
        , *args
        , **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.gt_paths = self._parse_root(gt_root, repeat_count, sort=True)
        self.transform = self._parse_transform(transform)
        self.device = torch.device(device)
        self.dtype = torch.dtype(dtype)
        self.scale_factor = scale_factor
        self.interpolation_mode = interpolation_mode

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, Number):
            idx = [idx]
        
        Y = torch.stack([self._load_img(self.gt_paths[i], dtype=self.dtype, device=device) for i in idx])
        Y = self.transform(Y)
        X = F.interpolate(Y, scale_factor=self.scale_factor, mode=self.interpolation_mode)
            
        if len(X) == 1 and len(Y) == 1:
            X = X[0]
            Y = Y[0]
            
        return X, Y
        