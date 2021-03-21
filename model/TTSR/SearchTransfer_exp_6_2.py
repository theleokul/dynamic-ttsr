import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rbf(x, y, gamma=1.):
    return torch.exp(-gamma * torch.linalg.norm(x - y) ** 2)


def rbf_bmm(x, y, gamma=1.):
    res = torch.empty(x.size(0), x.size(1), y.size(2), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)

    for b in range(res.size(0)):
        for i in range(res.size(1)):
            for j in range(res.size(2)):
                res[b, i, j] = rbf(x[b, i], y[b, :, j])

    return res


class SearchTransfer(nn.Module):
    def __init__(self, gamma=1.):
        super(SearchTransfer, self).__init__()
        self.gamma = gamma

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...] = [N, C*k*k, Hr*Wr]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]  # [N, 1, -1]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        # expanse shape: [-1, C*k*k, -1]
        index = index.view(views).expand(expanse)  # [N, H*W] -> [N, 1, H*W] -> [N, C*k*k, H*W]
        return torch.gather(input, dim, index)  # [N, C*k*k, H*W]

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        ### search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)  # [N, C*k*k, H*W]
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)  # [N, C*k*k, H*W]
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)  # [N, H*W, C*k*k]

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]

        # R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, H*W, H*W]
        R_lv3 = rbf_bmm(refsr_lv3_unfold, lrsr_lv3_unfold, self.gamma)  #[N, H*W, H*W]
        # R_lv3_star - SOFT ATTENTION MAP, R_lv3_star_arg - HARD ATTENTION MAP
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W]

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)  # [N, C*k*k, H*W]
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2) # [N, C*2k*2k/2, H*W]
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4) # # [N, C*4k*4k/4, H*W]

        # Get features from ref based on HARD ATTENTION MAP (patch selection with index)
        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)  # [N, C*k*k, H*W]
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)  # [N, C*2k*2k, H*W]
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)  # [N, C*4k*4k, H*W]

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        return S, T_lv3, T_lv2, T_lv1
