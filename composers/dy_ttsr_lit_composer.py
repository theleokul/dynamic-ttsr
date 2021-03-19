import sys
import pathlib
import math
import typing as T

import numpy as np
import cv2
import torch
import kornia as K

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))
from composers.full_ttsr_lit_composer import FullTTSRLitComposer
import composers.utils as cutils
from loss import SparsityLoss


class DyTTSRLitComposer(FullTTSRLitComposer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_num = self.kwargs['trainer__kwargs']['max_epochs']
        self.gumbel_temp = 1.0
        self.gumbel_noise = True

    def training_step(self, batch, batch_idx):
        lr = batch['LR']
        lr_sr = batch['LR_sr']
        hr = batch['HR']
        ref = batch['Ref']
        ref_sr = batch['Ref_sr']

        meta = {
            'masks_11': [], 'masks_21': [], 'masks_22': [], 'masks_31': [], 'masks_32': [], 'masks_33': []
            , 'device': self.device, 'gumbel_temp': self.gumbel_noise, 'gumbel_noise': self.gumbel_noise, 'epoch': self.current_epoch
        }

        sr, S, T_lv3, T_lv2, T_lv1, meta = self(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr, meta=meta)

        loss_rec = self.loss['rec'](sr, hr)
        if 'bsparse' is self.loss.losses:
            loss_sparse = self.loss['bsparse'](meta)

        loss = loss_rec * self.loss['coef_rec'] \
            + loss_sparse * self.loss['coef_bsparse']
        
        output = {
            'loss': loss
            , 'bs': loss_sparse.detach()
        }

        self.log_dict(output, prog_bar=True)

        return output

    def training_epoch_end(self, outputs):
        super().training_epoch_end(outputs)
        # disable gumbel noise in finetuning stage
        self.gumbel_noise = False if (self.current_epoch + 1) > 0.8 * self.epoch_num else True

    def validation_step(self, batch, batch_idx):
        lr = batch['LR']
        lr_sr = batch['LR_sr']
        hr = batch['HR']
        ref = batch['Ref']
        ref_sr = batch['Ref_sr']

        meta = {
            'masks_11': [], 'masks_21': [], 'masks_22': [], 'masks_31': [], 'masks_32': [], 'masks_33': []
            , 'device': self.device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': self.current_epoch
        }

        sr, _, _, _, _, meta = self(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr, meta=meta)
        psnr_loss, ssim_loss = cutils.calc_psnr_and_ssim(sr.detach(), hr.detach())

        output = {
            'psnr': psnr_loss
            , 'ssim': ssim_loss
        }

        return output

    def predict_step(self, batch, batch_idx, output_dirpath='./output', device=torch.device('cpu')):
        lr = batch['LR'].to(device)
        lr_sr = batch['LR_sr'].to(device)
        hr = batch['HR'].to(device)
        ref = batch['Ref'].to(device)
        ref_sr = batch['Ref_sr'].to(device)

        meta = {
            'masks_11': [], 'masks_21': [], 'masks_22': [], 'masks_31': [], 'masks_32': [], 'masks_33': []
            , 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': self.current_epoch
        }

        with torch.no_grad():
            sr, _, _, _, _, meta = self(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr, meta=meta)

        # viz.plot_image(input)
        # viz.plot_ponder_cost(meta['masks'])
        # viz.plot_masks(meta['masks'])
        # plt.show()

        x = lr
        y_pred = sr
        y = hr

        # Denormalize
        x = (x + 1.) * 127.5
        x = np.transpose(x.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)

        y_pred = (y_pred + 1.) * 127.5
        y_pred = np.transpose(y_pred.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)

        y = (y + 1.) * 127.5
        y = np.transpose(y.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        
        ref = (ref + 1.) * 127.5
        ref = np.transpose(ref.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)

        imageio.imsave(os.path.join(output_dirpath, f'{batch_idx}_input.png'), x)
        imageio.imsave(os.path.join(output_dirpath, f'{batch_idx}_pred.png'), y_pred)
        imageio.imsave(os.path.join(output_dirpath, f'{batch_idx}_gt.png'), y)
        imageio.imsave(os.path.join(output_dirpath, f'{batch_idx}_ref.png'), ref)
