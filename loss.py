import math
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
    def __init__(self, use_S=False, type='l2'):
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
            if T_lv2 is not None:
                loss_texture += F.l1_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            if T_lv1 is not None:
                loss_texture += F.l1_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= len(list(filter(lambda x: x is not None, [T_lv3, T_lv2, T_lv1])))
        elif (self.type == 'l2'):
            loss_texture  = F.mse_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            if T_lv2 is not None:
                loss_texture += F.mse_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            if T_lv1 is not None:
                loss_texture += F.mse_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= len(list(filter(lambda x: x is not None, [T_lv3, T_lv2, T_lv1])))
        
        return loss_texture


class WGANGPAdversarialLoss(nn.Module):            
    def forward(self, d_fake, d_real=None, d_hat=None, fake=None, real=None, hat=None, discriminator=False):
        if discriminator:
            loss_d = (d_fake - d_real).mean()
            gradients = torch.autograd.grad(
                outputs=d_hat.sum(), inputs=hat,
                retain_graph=True
                , create_graph=True
                , only_inputs=True
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


class SparsityLoss(nn.Module):
    ''' 
    Defines the sparsity loss, consisting of two parts:
    - network loss: MSE between computational budget used for whole network and target 
    - block loss: sparsity (percentage of used FLOPS between 0 and 1) in a block must lie between upper and lower bound. 
    This loss is annealed.
    '''

    def __init__(self, sparsity_target, num_epochs, logger=None):
        super().__init__()
        self.sparsity_target = sparsity_target
        self.num_epochs = num_epochs
        self.logger = logger

    def forward(self, meta):

        p = meta['epoch'] / (0.33*self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound = (1 - progress*(1-self.sparsity_target))
        lower_bound = progress*self.sparsity_target

        loss_block = torch.tensor(.0).to(device=meta['device'])
        cost, total = torch.tensor(.0).to(device=meta['device']), torch.tensor(.0).to(device=meta['device'])

        maskss = meta['masks_11'], meta['masks_21'], meta['masks_22'], meta['masks_31'], meta['masks_32'], meta['masks_33']
        for k, masks in enumerate(maskss):
            for i, mask in enumerate(masks):
                m_dil = mask['dilate']
                m = mask['std']

                c = m_dil.active_positions * m_dil.flops_per_position + \
                    m.active_positions * m.flops_per_position
                t = m_dil.total_positions * m_dil.flops_per_position + \
                    m.total_positions * m.flops_per_position

                layer_perc = c / t
                if self.logger is not None:
                    self.logger.log(f'layer_perc_{k}_'+str(i), layer_perc.item())

                assert layer_perc >= 0 and layer_perc <= 1, layer_perc
                loss_block += max(0, layer_perc - upper_bound)**2  # upper bound
                loss_block += max(0, lower_bound - layer_perc)**2  # lower bound

                cost += c
                total += t

        perc = cost/total
        assert perc >= 0 and perc <= 1, perc
        loss_block /= (len(meta['masks_11']) + len(meta['masks_21']) + \
                        len(meta['masks_22']) + len(meta['masks_31']) + \
                        len(meta['masks_32']) + len(meta['masks_33']))
        loss_network = (perc - self.sparsity_target)**2

        if self.logger is not None:
            self.logger.log_dict({
                'upper_bound': upper_bound
                , 'lower_bound': lower_bound
                , 'cost_perc': perc.item()
                , 'loss_sp_block': loss_block.item()
                , 'loss_sp_network': loss_network.item()
            })

        return loss_network + loss_block
