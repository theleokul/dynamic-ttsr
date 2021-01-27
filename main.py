import sys
import pathlib
import functools
import typing as T
import argparse

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
import lit_pan


parser = argparse.ArgumentParser(description='Main entry to train/evaluate models.')
parser.add_argument('--config', type=str)

args = parser.parse_args()
config = utils.parse_config(args.config)


if __name__ == "__main__":
    # Pop metainfo
    log_save_dir = config.get('log_save_dir', './logs')
    trainer__kwargs = config.get('trainer__kwargs', dict())
    model_checkpoint_callback__monitor = config.get('model_checkpoint_callback__monitor', 'avg_val_loss')
    model_checkpoint_callback__save_top_k = config.get('model_checkpoint_callback__save_top_k', 1)
    model_checkpoint_callback__mode = config.get('model_checkpoint_callback__mode', 'min')

    # TODO: Code abstract lit module to build different models here
    lit_model = lit_pan.LitPAN(**config)
    
    # Form logger and checkpoint callback (needed for every case)
    logger = pl.loggers.TensorBoardLogger(save_dir=log_save_dir, name='')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=model_checkpoint_callback_monitor
        , save_top_k=model_checkpoint_callback__save_top_k
        , mode=model_checkpoint_callback__mode
        , filepath=str(
            Path(
                logger.log_dir
                , 'checkpoints'
                , '{epoch}-{' + model_checkpoint_callback__monitor + ':.4f}'
            )
        )
    )
    trainer = pl.Trainer(
        **trainer__kwargs
        , checkpoint_callback=checkpoint_callback
        , logger=logger
    )
    trainer.fit(model)
