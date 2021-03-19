import os
import sys
import pathlib
import math

import numpy as np
import cv2
import torch
import kornia as K
import imageio

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
from base_lit_composer import BaseLitComposer
import utils


class TTSRLitComposer(BaseLitComposer):
    
    def training_step(self, batch, batch_idx):
        lr = batch['LR']
        lr_sr = batch['LR_sr']
        hr = batch['HR']
        ref = batch['Ref']
        ref_sr = batch['Ref_sr']

        sr, S, T_lv3, T_lv2, T_lv1 = self(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
        loss = self.loss(sr, hr)

        return loss

    def validation_step(self, batch, batch_idx):
        lr = batch['LR']
        lr_sr = batch['LR_sr']
        hr = batch['HR']
        ref = batch['Ref']
        ref_sr = batch['Ref_sr']

        sr, _, _, _, _ = self(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
        psnr_loss, ssim_loss = utils.calc_psnr_and_ssim(sr.detach(), hr.detach())

        output = {
            'psnr': psnr_loss
            , 'ssim': ssim_loss
        }

        return output

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, output_dirpath='./output', device=torch.device('cpu')):
        lr = batch['LR'].to(device)
        lr_sr = batch['LR_sr'].to(device)
        hr = batch['HR'].to(device)
        ref = batch['Ref'].to(device)
        ref_sr = batch['Ref_sr'].to(device)

        with torch.no_grad():
            sr, _, _, _, _ = self(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)

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
