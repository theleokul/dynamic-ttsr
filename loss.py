import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, sr, hr):
        return self.loss(sr, hr)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

    def forward(self, sr_relu5_1, hr_relu5_1):
        loss = F.mse_loss(sr_relu5_1, hr_relu5_1)
        return loss


class TPerceptualLoss(nn.Module):
    def __init__(self, use_S=True, type='l2'):
        super(TPerceptualLoss, self).__init__()
        self.use_S = use_S
        self.type = type

    def gram_matrix(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, h*w)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def forward(self, map_lv3, map_lv2, map_lv1, S, T_lv3, T_lv2, T_lv1):
        ### S.size(): [N, 1, h, w]
        if (self.use_S):
            S_lv3 = torch.sigmoid(S)
            S_lv2 = torch.sigmoid(F.interpolate(S, size=(S.size(-2)*2, S.size(-1)*2), mode='bicubic'))
            S_lv1 = torch.sigmoid(F.interpolate(S, size=(S.size(-2)*4, S.size(-1)*4), mode='bicubic'))
        else:
            S_lv3, S_lv2, S_lv1 = 1., 1., 1.

        if (self.type == 'l1'):
            loss_texture  = F.l1_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.l1_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            loss_texture += F.l1_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.
        elif (self.type == 'l2'):
            loss_texture  = F.mse_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.mse_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            loss_texture += F.mse_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.
        
        return loss_texture


class WGANGPAdversarialLoss(nn.Module):            
    def forward(self, d_fake, d_real=None, d_hat=None, fake=None, real=None, hat=None, discriminator=False):
        if discriminator:
            loss_d = (d_fake - d_real).mean()
            gradients = torch.autograd.grad(
                outputs=d_hat.sum(), inputs=hat,
                retain_graph=True, create_graph=True, only_inputs=True
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_norm = gradients.norm(2, dim=1)
            gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
            loss_d += gradient_penalty
        else:
            loss_d = -d_fake.mean()

        return loss_d


class DictBasedLoss:
    def __init__(self, coefs: T.Dict[str, float], losses: T.Dict[str, T.Callable]):
        self.coefs = coefs
        self.losses = losses

    def __getitem__(self, key: str):
        if key.startswith('coef_'):
            item = self.coefs[key[5:]]
        else:
            item =  self.losses[key]
        return item
