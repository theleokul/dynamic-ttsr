import sys
import pathlib

import torch
import kornia as K

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
from base_lit_composer import BaseLitComposer


class TTSRLitComposer(BaseLitComposer):
    
    def training_step(self, batch, batch_idx):
        lr = batch['LR']
        lr_sr = batch['LR_sr']
        hr = batch['HR']
        ref = batch['Ref']
        ref_sr = batch['Ref_sr']

        sr, S, T_lv3, T_lv2, T_lv1 = self(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
        loss = self.loss(sr, hr)
        self.log('loss_g', loss, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        lr = batch['LR']
        lr_sr = batch['LR_sr']
        hr = batch['HR']
        ref = batch['Ref']
        ref_sr = batch['Ref_sr']

        sr, _, _, _, _ = self(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
        psnr_loss = K.losses.psnr((sr+1.)/2, (hr+1.)/2, max_val=1.)
        ssim_loss = K.losses.ssim((sr+1.)/2, (hr+1.)/2, window_size=5).mean()

        output = {
            'val_psnr': psnr_loss
            , 'val_ssim': ssim_loss
        }
        self.log_dict(output, prog_bar=True, sync_dist=True)

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
            loss = self.loss['rec'](sr, hr) * self.loss['coef_rec']
            if 'per' in self.loss.losses:
                sr_relu5_1 = self.vgg19((sr + 1.) / 2.)
                with torch.no_grad():
                    hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)
                loss += self.loss['per'](sr_relu5_1, hr_relu5_1) * self.loss['coef_per']
            if 'tper' in self.loss.losses:
                sr_lv1, sr_lv2, sr_lv3 = self(sr=sr)
                loss += self.loss['tper'](sr_lv3, sr_lv2, sr_lv1, S, T_lv3, T_lv2, T_lv1) * self.loss['coef_tper']
            if 'adv' in self.loss.losses:
                fake = sr
                d_fake = self.discriminator(fake)
                loss += self.loss['adv'](d_fake, discriminator=False) * self.loss['coef_adv']
            self.log('loss_g', loss, sync_dist=True)

            return {
                'loss': loss
                , 'loss_g': loss
            }
        elif optimizer_idx == 1:  # Discriminator
            fake, real = sr, hr
            fake_detach = fake.detach()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            epsilon = torch.rand(real.size(0), 1, 1, 1).to(d_fake.device)
            epsilon = epsilon.expand(real.size())
            hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
            hat.requires_grad = True
            d_hat = self.discriminator(hat)
            loss = self.loss['adv'](d_fake, d_real, d_hat, fake, real, hat, discriminator=True)
            self.log('loss_d', loss, sync_dist=True)

            return {
                'loss': loss
                , 'loss_d': loss
            }
