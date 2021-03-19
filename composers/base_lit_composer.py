import sys
import pathlib
import functools
import typing as T
from types import MethodType
import importlib
import abc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
import kornia as K
import pytorch_lightning as pl

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH.parent))
from model import TTSR, Vgg19, Discriminator
from dataset import CufedTrainDataset, CufedTestDataset
from loss import ReconstructionLoss, PerceptualLoss, TPerceptualLoss, WGANGPAdversarialLoss, DictBasedLoss


class BaseLitComposer(pl.LightningModule, abc.ABC):

    @staticmethod
    def _parse_loss_atom(loss_name: str) -> T.Callable:
        loss = None
        if loss_name == 'rec':
            loss = ReconstructionLoss()
        elif loss_name == 'per':
            loss = PerceptualLoss()
        elif loss_name == 'tper':
            loss = TPerceptualLoss()
        elif loss_name == 'adv':
            loss = WGANGPAdversarialLoss()
        elif loss_name == 'bsparse':
            loss = SparsityLoss(-1, 2)  # no sparsity
        elif loss_name == 'sparse':
            loss = SparsityLoss(0.7, 50)

        return loss

    @classmethod
    def _parse_loss(cls, loss_components: str) -> T.Union[T.Callable, T.Dict]:
        components = loss_components.split('+')
        coefs = {}
        losses = {}

        for c in components:
            coef_loss = c.split('*')

            if len(coef_loss) < 2:
                coef, loss_atom_name = 1., coef_loss[0]
            else:
                coef, loss_atom_name = coef_loss
                coef = float(coef)
            
            loss = cls._parse_loss_atom(loss_atom_name)
            coefs[loss_atom_name] = coef
            losses[loss_atom_name] = loss

        if len(losses) > 1:
            loss = DictBasedLoss(coefs, losses)
        else:
            loss = list(losses.values())[0]
        
        return loss

    @staticmethod
    def _parse_model(model: str, *args, **kwargs) -> nn.Module:
        model = eval(model)(*args, **kwargs)
        return model

    @staticmethod
    def _parse_dataset(dataset: str, *args, **kwargs) -> data.Dataset:
        dataset = eval(dataset)(*args, **kwargs)
        return dataset

    @staticmethod
    def _parse_lr_scheduler(lr_scheduler: T.Optional[str], optimizer: optim.Optimizer, *args, **kwargs):
        if lr_scheduler is not None:
            scheduler = getattr(optim.lr_scheduler, lr_scheduler)(
                optimizer
                , *args, **kwargs
            )
        else:
            scheduler = None
        return scheduler

    def _parse_optimizer(self, optimizer: str, model: nn.Module, lr: T.Union[float, T.Dict[str, float]], *args, **kwargs) -> optim.Optimizer:
        if isinstance(lr, dict):
            parameters = []
            for k in lr:
                parameters.append({
                    'params': filter(lambda p: p.requires_grad, getattr(model, k).parameters())
                    , 'lr': lr[k]
                })
        else:
            parameters = [{
                'params': filter(lambda p: p.requires_grad, model.parameters())
                , 'lr': lr
            }]

        optimizer = getattr(optim, optimizer)(parameters, *args, **kwargs)

        return optimizer

    def __init__(
        self

        # Training
        , batch_size: int=1
        , loss: str='rec'
        
        # Model
        , model: str='TTSR'
        , model__args: T.List=[]
        , model__kwargs: T.Dict={}

        , optimizer: str='Adam'
        , lr: T.Union[float, T.Dict[str, float]]=1e-3
        , optimizer__args: T.List=[]
        , optimizer__kwargs: T.Dict={
            'betas': [0.9, 0.99]
            , 'weight_decay': 0
        }

        , lr_scheduler: T.Optional[str]=None
        , lr_scheduler__args: T.List=[]
        , lr_scheduler__kwargs: T.Dict={}

        # Discriminator
        , use_discriminator: bool=False

        , discriminator: str='Discriminator'
        , discriminator__args: T.List=[]
        , discriminator__kwargs: T.Dict={}

        , discriminator__optimizer: str='Adam'
        , discriminator__lr: float=1e-3
        , discriminator__optimizer__args: T.List=[]
        , discriminator__optimizer__kwargs: T.Dict={
            'betas': [0., 0.9]
            , 'weight_decay': 0
        }

        , discriminator__frequency: T.Optional[int]=None
        , discriminator__lr_scheduler: T.Optional[str]=None
        , discriminator__lr_scheduler__args: T.List=[]
        , discriminator__lr_scheduler__kwargs: T.Dict={}

        , add_vgg19: bool=False

        , *args
        , **kwargs
    ):
        super().__init__()

        self.batch_size = batch_size
        self.loss = self._parse_loss(loss)

        self.model = self._parse_model(model, *model__args, **model__kwargs)
        self.optimizer = self._parse_optimizer(optimizer, self.model, lr, *optimizer__args, **optimizer__kwargs)
        self.lr_scheduler = self._parse_lr_scheduler(lr_scheduler, self.optimizer, *lr_scheduler__args, **lr_scheduler__kwargs)

        self.use_discriminator = use_discriminator
        if self.use_discriminator:
            self.discriminator = self._parse_model(discriminator, *discriminator__args, **discriminator__kwargs)
            self.discriminator__optimizer = self._parse_optimizer(discriminator__optimizer, self.discriminator, discriminator__lr, *discriminator__optimizer__args, **discriminator__optimizer__kwargs)
            self.discriminator__frequency = discriminator__frequency
            self.discriminator__lr_scheduler = self._parse_lr_scheduler(discriminator__lr_scheduler, self.discriminator__optimizer, *discriminator__lr_scheduler__args, **discriminator__lr_scheduler__kwargs)

        if add_vgg19:
            self.vgg19 = Vgg19.Vgg19(requires_grad=False)

        # Basically some meta info to save within logger
        self.args = args
        self.kwargs = kwargs

        self.save_hyperparameters()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizers = []
        generator_optimizer = {'optimizer': self.optimizer}
        if self.lr_scheduler is not None:
            generator_optimizer['scheduler'] = self.lr_scheduler
        optimizers.append(generator_optimizer)

        if self.use_discriminator:
            discriminator_optimizer = {'optimizer': self.discriminator__optimizer}
            if self.discriminator__lr_scheduler is not None:
                discriminator_optimizer['scheduler'] = self.discriminator__lr_scheduler
            if self.discriminator__frequency is not None:
                discriminator_optimizer['frequency'] = self.discriminator__frequency
                generator_optimizer['frequency'] = 1
            optimizers.append(discriminator_optimizer)

        return optimizers

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def training_epoch_end(self, outputs):
        keys = set()
        # NOTE: Check keys only in first 5 outputs by default (Override if more needed)
        for o in outputs[:2]:
            keys |= set(o.keys())
        
        to_log = {
            f'a_{k}': \
            torch.as_tensor([o[k] for o in outputs if o.get(k, None) is not None]).mean() \
            for k in keys
        }
        self.log_dict(to_log, prog_bar=True)

    def validation_epoch_end(self, outputs):
        keys = set()
        # NOTE: Check keys only in first 5 outputs by default (Override if more needed)
        for o in outputs[:2]:
            keys |= set(o.keys())
        
        to_log = {
            f'a_{k}': \
            torch.as_tensor([o[k] for o in outputs if o.get(k, None) is not None]).mean() \
            for k in keys
        }
        self.log_dict(to_log, prog_bar=True)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)
