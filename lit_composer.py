import sys
import pathlib
import functools
import typing as T
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
import kornia as K
import kornia.augmentation as KA
import pytorch_lightning


class LitComposer(pl.LightningModule):

    @staticmethod
    def _parse_loss(loss: str) -> T.Callable:
        loss = loss.lower()
        if loss = 'l1':
            loss = F.l1_loss
        elif loss = 'l2':
            loss = F.mse_loss
        else:
            raise NotImplementedError()
        return loss

    @staticmethod
    def _parse_step(package_path, module_name, step_function):
        sys.path.append(step__package_path)
        step_module = importlib.import_module(module_name, package_path)
        step_function = getattr(step_module, step_function)
        return step_function

    @staticmethod
    def _parse_model(package_path: str, module_name: str, model_class: str, *args, **kwargs) -> nn.Module:
        sys.path.append(package_path)
        model_module = importlib.import_module(module_name, package_path)
        model_class = getattr(model_module, model_class)
        model = model_class(*args, **kwargs)
        return model

    @staticmethod
    def _parse_dataset(package_path: str, module_name: str, dataset_class: str, *args, **kwargs) -> data.Dataset:
        sys.path.append(package_path)
        dataset_module = importlib.import_module(module_name, package_path)
        dataset_class = getattr(dataset_module, dataset_class)
        dataset = dataset_class(*args, **kwargs)
        return dataset

    @staticmethod
    def _parse_lr_scheduler(lr_scheduler: str, optimizer: optim.optimizer.Optimizer, *args, **kwargs) -> optim._LRScheduler:
        return getattr(optim.lr_scheduler, lr_scheduler)(
            optimizer
            , *args, **kwargs
        )

    def _parse_optimizer(self, optimizer: str, *args, **kwargs) -> optim.optimizer.Optimizer:
        return getattr(optim, optimizer)(
            filter(lambda p: p.requires_grad, self.model.parameters())
            , *args, **kwargs
        )

    def __init__(
        self,

        # Training
        , batch_size: int=1
        , loss: str='l1'
        , mode: str='train+val'  # Combination of train, val and test

        , step__package_path: T.Optional[str]=None
        , step__module: str='step'
        , train_step__function: str='train_step'
        , validation_step__function: str='validation_step'
        , test_step__function: str='test_step'

        , optimizer: str='Adam'
        , optimizer__args: T.List=[]
        , optimizer__kwargs: T.Dict={
            'betas': [0.9, 0.99]
            , 'weight_decay': 0
            , 'lr': 1e-3
        }

        , lr_scheduler: str='CosineAnnealingWarmRestarts'
        , lr_scheduler__args: T.List=[]
        , lr_scheduler__kwargs: T.Dict={
            'T_0': 250e3
            , 'T_mult': 1
            , 'eta_min': 1e-7
        }
        
        # Model
        , model__package_path: str='./model'
        , model__module: str='resnet'
        , model__class: str='ResNet'
        , model__args: T.List=[]
        , model__kwargs: T.Dict={}

        # Dataset
        , dataset__package_path: str='./dataset'
        , dataset__module: str='lr_hr_dataset'
        , dataset__class: str='LRHRDataset'

        , train_dataset__args: T.List=[]
        , train_dataset__kwargs: T.Dict={}
        , train_num_workers: int=0

        , val_dataset__args: T.List=[]
        , val_dataset__kwargs: T.Dict={}
        , val_num_workers: int=0

        , test_dataset__args: T.List=[]
        , test_dataset__kwargs: T.Dict={}
        , test_num_workers: int=0

        , *args
        , **kwargs
    ):

        super().__init__()

        self.batch_size = batch_size
        self.loss = self._parse_loss(loss)

        if step__package_path is not None:
            self.train_step = self._parse_step(step__package_path, step__module, train_step__function)
            self.validation_step = self._parse_step(step__package_path, step__module, validation_step__function)
            self.test_step = self._parse_step(step__package_path, step__module, test_step__function)

        self.model = self._parse_model(model__package_path, model__module, model__class, *model__args, **model__kwargs)
        self.optimizer = self._parse_optimizer(optimizer, *optimizer__args, **optimizer__kwargs)
        self.lr_scheduler = self._parse_lr_scheduler(lr_scheduler, self.optimizer, *lr_scheduler__args, **lr_scheduler__kwargs)

        self.modes = mode.split('+')
        if 'train' in self.modes:
            self.train_dataset = self._parse_dataset(dataset__package_path, dataset__module, dataset__class, *train_dataset__args, **train_dataset__kwargs)
        if 'val' in self.modes:
            self.val_dataset = self._parse_dataset(dataset__package_path, dataset__module, dataset__class, *val_dataset__args, **val_dataset__kwargs)
        if 'test' in self.modes:
            self.test_dataset = self._parse_dataset(dataset__package_path, dataset__module, dataset__class, *test_dataset__args, **test_dataset__kwargs)

        # Basically some meta info to save within logger
        self.args = args
        self.kwargs = kwargs

        self.save_hyperparameters()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        if 'train' in self.modes:
            dl = data.DataLoader(
                self.train_dataset
                , batch_size=self.batch_size
                , shuffle=True
                , num_workers=self.train_num_workers
            )
        else:
            dl = super().train_dataloader()
        return dl

    def val_dataloader(self):
        if 'val' in self.modes:
            dl = data.DataLoader(
                self.val_dataset
                , batch_size=self.batch_size
                , num_workers=self.train_num_workers
            )
        else:
            dl = super().val_dataloader()
        return dl

    def test_dataloader(self):
        if 'test' in self.modes:
            dl = data.DataLoader(
                self.test_dataset
                , batch_size=self.batch_size
                , num_workers=self.train_num_workers
            )
        else:
            dl = super().test_dataloader()
        return dl

    def training_epoch_end(self, outputs):
        output = {f'avg_{k}': torch.as_tensor([o[k] for o in outputs]).mean() for k in outputs[0].keys()}
        self.log_dict(output, prog_bar=True)

    def validation_epoch_end(self, outputs):
        self.training_epoch_end(outputs)

    def test_epoch_end(self, outputs):
        self.training_epoch_end(outputs)

    def train_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        psnr_loss = K.losses.psnr(y_hat, y, max_val=1.)
        ssim_loss = K.losses.ssim(y_hat, y, window_size=5).mean()

        output = {
            'val_loss': loss
            , 'val_psnr': psnr_loss
            , 'val_ssim': ssim_loss
        }
        self.log_dict(output, prog_bar=True)

        return output

    def test_step(self, batch, batch_idx):
        return self._validation_step(batch, batch_idx)
