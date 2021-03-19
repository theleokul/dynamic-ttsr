import sys
import pathlib
import math

import numpy as np
import cv2
import torch
import kornia as K

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))
from composers.full_ttsr_lit_composer import FullTTSRLitComposer
from composers.dy_ttsr_lit_composer import DyTTSRLitComposer
import composers.utils as cutils


class DyFullTTSRLitComposer(DyTTSRLitComposer):

    def training_step(self, batch, batch_idx, optimizer_idx):
        lr = batch['LR']
        lr_sr = batch['LR_sr']
        hr = batch['HR']
        ref = batch['Ref']
        ref_sr = batch['Ref_sr']

        meta = {
            'masks_11': [], 'masks_21': [], 'masks_22': [], 'masks_31': [], 'masks_32': [], 'masks_33': []
            , 'device': self.device, 'gumbel_temp': self.gumbel_temp, 'gumbel_noise': self.gumbel_noise, 'epoch': self.current_epoch
        }

        sr, S, T_lv3, T_lv2, T_lv1, meta = self(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr, meta=meta)

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
            if 'sparse' in self.loss.losses:
                loss_sparse = self.loss['sparse'](meta)

            loss = loss_rec * self.loss['coef_rec'] \
                + loss_per * self.loss['coef_per'] \
                + loss_tper * self.loss['coef_tper'] \
                + loss_adv * self.loss['coef_adv'] \
                + loss_sparse * self.loss['coef_sparse']
            
            output = {
                'loss': loss
                , 'g': loss.detach()
                , 'r': loss_rec.detach()
                , 'p': loss_per.detach()
                , 'tp': loss_tper.detach()
                , 'a': loss_adv.detach()
                , 's': loss_sparse.detach()
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
        to_log = {}
        for oouts in outputs:
            keys = oouts[0].keys()
            to_log.update({
                f'a_{k}': torch.as_tensor([
                    o[k] for o in oouts if o.get(k, None) is not None
                ]).mean() for k in keys
            })
        
        self.log_dict(to_log, prog_bar=True)
