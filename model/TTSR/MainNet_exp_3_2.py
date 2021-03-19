import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
from MainNet import conv1x1, conv3x3, ResBlock, SFE


"""
    Ablation study: only one scaling (instead of original 3)
"""


class Tail(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv_tail1 = conv3x3(n_feats, n_feats//2)
        self.conv_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, x3):
        x = self.conv_tail1(x3)
        x = self.conv_tail2(x)
        x = torch.clamp(x, -1, 1)
        
        return x


class MainNet(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(MainNet, self).__init__()
        self.num_res_blocks = num_res_blocks ### a list containing number of resblocks of different stages
        self.n_feats = n_feats

        self.SFE = SFE(self.num_res_blocks[0], n_feats, res_scale)

        ### stage11
        self.conv11_head = conv3x3(256+n_feats, n_feats)
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))
        self.conv11_tail = conv3x3(n_feats, n_feats)

        ### subpixel 1 -> 3
        self.conv13 = conv3x3(n_feats, n_feats*16)
        self.ps13 = nn.PixelShuffle(4)

        ### stage31, 33
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))

        self.conv33_tail = conv3x3(n_feats, n_feats)

        self.tail = Tail(n_feats)

    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None):
        ### shallow feature extraction
        x = self.SFE(x)

        ### stage11
        x11 = x

        ### soft-attention
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res)
        x11_res = x11_res * S
        x11 = x11 + x11_res

        x11_res = x11

        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        ### stage31, 33
        x31 = x11
        x31_res = x31

        # Upscale x1 to x3
        x33 = self.conv13(x11)
        x33 = F.relu(self.ps13(x33))
        
        x33_res = x33

        for i in range(self.num_res_blocks[2]):
            x33_res = self.RB33[i](x33_res)

        x33_res = self.conv33_tail(x33_res)

        x33 = x33 + x33_res

        x = self.tail(x33)

        return x
