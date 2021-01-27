import sys
import pathlib
import functools
import typing as T

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv
import kornia as K
import kornia.augmentation as KA
import pytorch_lightning as pl

DIR_PATH = pathlib.Path(__file__).resolve().parent()
sys.path.append(DIR_PATH)
import utils
import _pan as pan
import image2image_dataset as i2i_dataset


class LitPAN(pl.LightningModule, pan.PAN):

    @staticmethod
    def _parse_loss(loss: str) -> T.Callable:
        loss = loss.lower()
        if loss == 'l1':
            loss = F.l1_loss
        elif loss == 'l2':
            loss = F.mse_loss
        else:
            raise NotImplementedError()

        return loss

    def __init__(
        self

        # Training
        , loss: str='l1'
        , lr_min: float=1e-7
        , lr_max: float=1e-3
        , betas: T.Tuple[float, float]=[0.9, 0.99]
        , weight_decay: float=0
        , T_0: int=250e3
        , T_mult: int=1
        , batch_size: int=32

        # Model
        , model__args: T.List=[]
        , model__kwargs: T.Dict=dict()

        # Dataset
        , train_dataset__args: T.List=[]
        , train_dataset__kwargs: T.Dict=dict()
        , val_dataset__args: T.List=[]
        , val_dataset__kwargs: T.Dict=dict()

        , *other_args
        , **other_kwargs
    ):

        super(pl.LightningModule, self).__init__()
        super(pan.PAN, self).__init__(*model__args, **model__kwargs)

        # Training params
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.betas = betas
        self.weight_decay = weight_decay
        self.T_0 = T_0
        self.T_mult = T_mult
        self.batch_size = batch_size
        self.data_dirpath = data_dirpath
        self.loss = self._parse_loss(loss)
        
        # NOTE: dataset params will be parsed in train_dataloader, val_dataloader
        self.train_dataset__args = train_dataset__args
        self.train_dataset__kwargs = train_dataset__kwargs
        self.val_dataset__args = val_dataset__args
        self.val_dataset__kwargs = val_dataset__kwargs

        # NOTE: Basically not used, but useful for save_hyperparameters (params from main)
        self.other_args = other_args
        self.other_kwargs = other_kwargs
        
        self.save_hyperparameters(vars(self))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        losses = torch.as_tensor([o['loss'] for o in outputs])
        self.log('avg_train_loss', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        psnr_loss = K.losses.psnr(y_hat, y, max_val=1.)
        ssim_loss = K.losses.ssim(y_hat, y, window_size=5)

        output = {
            'val_loss': loss
            , 'val_psnr': psnr_loss
            , 'val_ssim': ssim_loss
        }
        self.log_dict(output, prog_bar=True)

        return output

    def validation_epoch_end(self, outputs):
        losses = []
        psnrs = []
        ssims = []

        for o in outputs:
            losses.append(o['val_loss'])
            psnrs.append(o['val_psnr'])
            ssims.append(o['val_ssim'])

        output = {
            'avg_val_loss': torch.as_tensor(losses).mean()
            , 'avg_val_psnr': torch.as_tensor(psnrs).mean()
            , 'avg_val_ssim': torch.as_tensor(ssims).mean()
        }
        self.log_dict(output, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters())
            , lr=self.lr_max
            , betas=self.betas
            , weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer
            , T_0=self.T_0
            , T_mult=self.T_mult
            , eta_min=self.lr_min
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = i2i_dataset.Image2ImageDataset(
            *self.train_dataset__args
            , **self.train_dataset__kwargs
        )

        train_loader = data.DataLoader(
            train_dataset
            , batch_size=self.batch_size
            , shuffle=True
        )

        return train_loader

    def val_dataloader(self):
        val_dataset = i2i_dataset.Image2ImageDataset(
            *self.val_dataset__args
            , **self.val_dataset__kwargs
        )

        val_loader = data.DataLoader(
            val_dataset
            , batch_size=self.batch_size
        )

        return val_loader
