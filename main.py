import os
import sys
import pathlib
import functools
import typing as T
import argparse
import warnings

from tqdm import tqdm
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as tv_transforms
import kornia as K
import kornia.augmentation as KA
import pytorch_lightning as pl

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(DIR_PATH)
import utils
import lit_composer


warnings.filterwarnings('ignore')  # To shut down some useless shit pytorch sometimes spits out
parser = argparse.ArgumentParser(description='Main entry to train/evaluate models.')
parser.add_argument('--config', nargs='+', type=str, required=True)
parser.add_argument('-g', '--gpus', nargs='+', type=int, default=[0], help='GPUs')

args = parser.parse_args()
config = {}
for c in args.config:
    config_ = utils.parse_config(c)
    config.update(config_)
gpus = args.gpus


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(g) for g in gpus])

    # Pop metainfo
    modes = config.get('modes', 'train+val').split('+')
    output_dirpath = config.get('output_dir', './output')
    log_save_dir = config.get('log_save_dir', './logs')
    trainer__kwargs = config.get('trainer__kwargs', {})
    model_checkpoint = config.get('model_checkpoint', None)
    model_checkpoint_callback__monitor = config.get('model_checkpoint_callback__monitor', ['avg_val_loss'])
    model_checkpoint_callback__save_top_k = config.get('model_checkpoint_callback__save_top_k', 1)
    model_checkpoint_callback__mode = config.get('model_checkpoint_callback__mode', 'min')

    if model_checkpoint is not None:
        lit_model = lit_composer.LitComposer.load_from_checkpoint(model_checkpoint, **config)
    else:
        lit_model = lit_composer.LitComposer(**config)
    
    # Form logger and checkpoint callback
    if 'train' in modes:
        logger = plt.loggers.TensorBoardLogger(save_dir=log_save_dir, name='')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=model_checkpoint_callback__monitor[0]
            , save_top_k=model_checkpoint_callback__save_top_k
            , mode=model_checkpoint_callback__mode
            , save_weights_only=True
            , filepath=str(
                pathlib.Path(
                    logger.log_dir
                    , 'checkpoint'
                    , '{epoch}-' + '-'.join(['{' + m + ':.4f' for m in model_checkpoint_callback__monitor])
                )
            )
        )
        trainer = pl.Trainer(
            **trainer__kwargs
            , checkpoint_callback=checkpoint_callback
            , logger=logger
            , gpus=len(gpus)
        )
        trainer.fit(lit_model)
    elif 'test' in modes:
        lit_model.eval()
        trainer = pl.Trainer(
            **trainer__kwargs
            , gpus=len(gpus)
        )
        trainer.test(lit_model, verbose=True)
    elif 'predict' in modes:
        lit_model.freeze()
        os.makedirs(output_dirpath, exist_ok=True)

        # Deploy model on the appropriate device
        device = torch.device('cuda') if len(gpus) >= 1 else torch.device('cpu')
        lit_model.to(device)

        to_PIL_transformer = tv_transforms.TOPILImage()
        dataloader = lit_model.test_dataloader()
        for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = x.to(device)
            y = y.to(device)

            y_pred = lit_model(x)
            if not isinstance(y_pred, torch.Tensor):
                # In case model spits out several objects
                y_pred = y_pred[0]

            for j, (x, yp, y) in enumerate(zip(x, y_pred, y)):
                k = i * lit_model.batch_size + j
                to_PIL_transformer(x).save(os.path.join(output_dirpath, f'{k}_input.png'))
                to_PIL_transformer(yp).save(os.path.join(output_dirpath, f'{k}_pred.png'))
                to_PIL_transformer(y).save(os.path.join(output_dirpath, f'{k}_gt.png'))
    else:
        raise NotImplementedError()
