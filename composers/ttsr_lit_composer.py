import sys
import pathlib
import math

import numpy as np
import cv2
import torch
import kornia as K

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
from base_lit_composer import BaseLitComposer


def calc_psnr(img1, img2):
    ### args:
    # img1: [h, w, c], range [0, 255]
    # img2: [h, w, c], range [0, 255]
    diff = (img1 - img2) / 255.0
    diff[:,:,0] = diff[:,:,0] * 65.738 / 256.0
    diff[:,:,1] = diff[:,:,1] * 129.057 / 256.0
    diff[:,:,2] = diff[:,:,2] * 25.064 / 256.0

    diff = np.sum(diff, axis=2)
    mse = np.mean(np.power(diff, 2))
    return -10 * math.log10(mse)
    
def calc_ssim(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

        ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
        # the same outputs as MATLAB's
    border = 0
    img1_y = np.dot(img1, [65.738,129.057,25.064])/256.0+16.0
    img2_y = np.dot(img2, [65.738,129.057,25.064])/256.0+16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h-border, border:w-border]
    img2_y = img2_y[border:h-border, border:w-border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calc_psnr_and_ssim(sr, hr):
    ### args:
        # sr: pytorch tensor, range [-1, 1]
        # hr: pytorch tensor, range [-1, 1]

    ### prepare data
    sr = (sr+1.) * 127.5
    hr = (hr+1.) * 127.5
    if (sr.size() != hr.size()):
        h_min = min(sr.size(2), hr.size(2))
        w_min = min(sr.size(3), hr.size(3))
        sr = sr[:, :, :h_min, :w_min]
        hr = hr[:, :, :h_min, :w_min]

    img1 = np.transpose(sr.squeeze().round().cpu().numpy(), (1,2,0))
    img2 = np.transpose(hr.squeeze().round().cpu().numpy(), (1,2,0))

    psnr = calc_psnr(img1, img2)
    ssim = calc_ssim(img1, img2)

    return psnr, ssim


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
        # psnr_loss = K.losses.psnr((sr+1.)/2, (hr+1.)/2, max_val=1.)
        # ssim_loss = K.losses.ssim((sr+1.)/2, (hr+1.)/2, window_size=5).mean()
        # psnr_loss = K.losses.psnr(sr, hr, max_val=1.)
        # ssim_loss = K.losses.ssim(sr, hr, window_size=5).mean()
        psnr_loss, ssim_loss = calc_psnr_and_ssim(sr.detach(), hr.detach())

        output = {
            'psnr': psnr_loss
            , 'ssim': ssim_loss
        }

        return output

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


class FullTTSRLitComposer(TTSRLitComposer):

    def load_baseline(self, baseline):
        self.model.load_state_dict(baseline.model.state_dict())

    def training_step(self, batch, batch_idx, optimizer_idx):
        lr = batch['LR']
        lr_sr = batch['LR_sr']
        hr = batch['HR']
        ref = batch['Ref']
        ref_sr = batch['Ref_sr']
        sr, S, T_lv3, T_lv2, T_lv1 = self(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)

        if optimizer_idx == 0:  # Generator
            loss_rec = self.loss['rec'](sr, hr)
            if 'per' in self.loss.losses:
                sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                with torch.no_grad():
                    hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                loss_per = self.loss['per'](sr_relu5_1, hr_relu5_1)
            if 'tper' in self.loss.losses:
                sr_lv1, sr_lv2, sr_lv3 = self(sr=sr)
                loss_tper = self.loss['tper'](sr_lv3, sr_lv2, sr_lv1, S, T_lv3, T_lv2, T_lv1)
            if 'adv' in self.loss.losses:
                fake = sr
                d_fake = self.discriminator(fake)
                loss_adv = self.loss['adv'](d_fake, discriminator=False)

            loss = loss_rec * self.loss['coef_rec'] + loss_per * self.loss['coef_per'] + loss_tper * self.loss['coef_tper'] + loss_adv * self.loss['coef_adv']
            
            output = {
                'loss': loss
                , 'g': loss.detach()
                , 'r': loss_rec.detach()
                , 'p': loss_per.detach()
                , 'tp': loss_tper.detach()
                , 'a': loss_adv.detach()
            }

        elif optimizer_idx == 1:  # Discriminator
            fake, real = sr, hr
            fake_detach = fake.detach()

            loss = 0
            for _ in range(2):
                d_fake = self.discriminator(fake_detach)
                d_real = self.discriminator(real)
                epsilon = torch.rand(real.size(0), 1, 1, 1).to(d_fake.device)
                epsilon = epsilon.expand(real.size())
                hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                hat.requires_grad = True
                d_hat = self.discriminator(hat)
                loss += self.loss['adv'](d_fake, d_real, d_hat, fake, real, hat, discriminator=True)

            output = {'loss': loss, 'd': loss.detach()}

        return output

    def training_epoch_end(self, outputs):
        keys = ['g', 'r', 'p', 'tp', 'a', 'd']
        to_log = {f'a_{k}': torch.as_tensor([o[k] for o in outputs]).mean() for k in keys}
        
        self.log_dict(to_log, prog_bar=True)
