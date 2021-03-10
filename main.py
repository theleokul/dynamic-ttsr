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
import torch.utils.data as data
import torchvision as tv
import torchvision.transforms as tv_transforms
# import kornia as K
# import kornia.augmentation as KA
import pytorch_lightning as pl

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(DIR_PATH)
import utils
# import lit_composer
import composers


warnings.filterwarnings('ignore')  # To shut down some useless shit pytorch sometimes spits out
parser = argparse.ArgumentParser(description='Main entry to train/evaluate models.')
parser.add_argument('--config', nargs='+', type=str, required=True)
parser.add_argument('-g', '--gpus', nargs='+', type=int, default=[], help='GPUs')

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

    Composer = getattr(composers, config.get('composer'))
    if model_checkpoint is not None:
        lit_model = Composer.load_from_checkpoint(model_checkpoint, **config)
    else:
        lit_model = Composer(**config)

    baseline_checkpoint = config.get('baseline_checkpoint', None)
    if baseline_checkpoint is not None:
        BComposer = getattr(composers, config.get('baseline_composer'))
        baseline_lit_model = BComposer.load_from_checkpoint(baseline_checkpoint)
        lit_model.load_baseline(baseline_lit_model)
        del baseline_lit_model
    
    # Form logger and checkpoint callback
    if 'train' in modes:
        logger = pl.loggers.TensorBoardLogger(save_dir=log_save_dir, name='')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=model_checkpoint_callback__monitor[0]
            , save_top_k=model_checkpoint_callback__save_top_k
            , mode=model_checkpoint_callback__mode
            , save_weights_only=False  # It allows to checkpoint generator, discriminator and all stuff
            , filename='{epoch}-' + '-'.join(['{' + m + ':.4f' + '}' for m in model_checkpoint_callback__monitor])
        )
        trainer = pl.Trainer(
            **trainer__kwargs
            , checkpoint_callback=checkpoint_callback
            , logger=logger
            , gpus=len(gpus)
        )

        train_dataset = lit_model._parse_dataset(
            config.get('train_dataset')
            , *config.get('train_dataset__args', [])
            , **config.get('train_dataset__kwargs', {})
        )
        train_data_loader = data.DataLoader(
            train_dataset
            , batch_size=config.get('batch_size', 1)
            , shuffle=True
            , num_workers=config.get('train_num_workers', 0)
        )
        val_dataset = lit_model._parse_dataset(
            config.get('val_dataset')
            , *config.get('val_dataset__args', [])
            , **config.get('val_dataset__kwargs', {})
        )
        val_data_loader = data.DataLoader(
            val_dataset
            , batch_size=1
            , num_workers=config.get('val_num_workers', 0)
        )        
        trainer.fit(lit_model, train_data_loader, val_data_loader)
    elif 'test' in modes:
        lit_model.eval()
        trainer = pl.Trainer(
            **trainer__kwargs
            , gpus=len(gpus)
        )

        test_dataset = lit_model._parse_dataset(
            config.get('test_dataset')
            , *config.get('test_dataset__args', [])
            , **config.get('test_dataset__kwargs', {})
        )
        test_data_loader = data.DataLoader(
            test_dataset
            , batch_size=config.get('batch_size', 1)
            , num_workers=config.get('test_num_workers', 0)
        )

        trainer.test(lit_model, test_data_loader)
    elif 'predict' in modes:
        lit_model.eval()
        os.makedirs(output_dirpath, exist_ok=True)

        # Deploy model on the appropriate device
        device = torch.device('cuda') if len(gpus) >= 1 else torch.device('cpu')
        lit_model.to(device)

        to_PIL_transformer = tv_transforms.ToPILImage()

        test_dataset = lit_model._parse_dataset(
            config.get('test_dataset')
            , *config.get('test_dataset__args', [])
            , **config.get('test_dataset__kwargs', {})
        )
        test_data_loader = data.DataLoader(
            test_dataset
            , batch_size=config.get('batch_size', 1)
            , num_workers=config.get('test_num_workers', 0)
        )

        for i, batch in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            lr = batch['LR'].to(device)
            lr_sr = batch['LR_sr'].to(device)
            hr = batch['HR'].to(device)
            ref = batch['Ref'].to(device)
            ref_sr = batch['Ref_sr'].to(device)
            sr, S, T_lv3, T_lv2, T_lv1 = lit_model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)

            x = lr
            y_pred = sr
            y = hr

            # Denormalize
            x = (x + 1.) * 127.5
            x = x.clamp(0., 255.).type(torch.uint8)
            y_pred = (y_pred + 1.) * 127.5
            y_pred = y_pred.clamp(0., 255.).type(torch.uint8)
            y = (y + 1.) * 127.5
            y = y.clamp(0., 255.).type(torch.uint8)
            ref = (ref + 1.) * 127.5
            ref = ref.clamp(0., 255.).type(torch.uint8)

            for j, (x, yp, y, ref) in enumerate(zip(x, y_pred, y, ref)):
                k = i * lit_model.batch_size + j
                to_PIL_transformer(x).save(os.path.join(output_dirpath, f'{k}_input.png'))
                to_PIL_transformer(yp).save(os.path.join(output_dirpath, f'{k}_pred.png'))
                to_PIL_transformer(y).save(os.path.join(output_dirpath, f'{k}_gt.png'))
                to_PIL_transformer(ref).save(os.path.join(output_dirpath, f'{k}_ref.png'))
    else:
        raise NotImplementedError()
