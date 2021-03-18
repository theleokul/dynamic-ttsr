import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
from MainNet import conv1x1, conv3x3, ResBlock, SFE


"""
    Ablation study: 2 number of scalings (instead of original 3)
"""


class CSFI13(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, 4)

        self.conv_merge1 = conv3x3(n_feats*2, n_feats)
        self.conv_merge2 = conv3x3(n_feats*2, n_feats)

    def forward(self, x1, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv12(x13))
        x31 = F.relu(self.conv21(x3))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x31), dim=1) ))
        x3 = F.relu(self.conv_merge2( torch.cat((x3, x13), dim=1) ))

        return x1, x3


class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats*2, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats//2)
        self.conv_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, x1, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x = F.relu(self.conv_merge( torch.cat((x3, x13), dim=1) ))
        x = self.conv_tail1(x)
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
        self.conv33_head = conv3x3(64+n_feats, n_feats)

        self.ex13 = CSFI13(n_feats)

        self.RB31 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        self.merge_tail = MergeTail(n_feats)

    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None):
        ### shallow feature extraction
        x = self.SFE(x)

        ### stage11
        x11 = x

        ### soft-attention
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
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

        ### soft-attention
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res)
        x33_res = x33_res * F.interpolate(S, scale_factor=4, mode='bicubic')
        x33 = x33 + x33_res
        
        x33_res = x33

        x31_res, x33_res = self.ex13(x31_res, x33_res)

        for i in range(self.num_res_blocks[2]):
            x31_res = self.RB31[i](x31_res)
            x33_res = self.RB33[i](x33_res)

        x31_res = self.conv31_tail(x31_res)
        x33_res = self.conv33_tail(x33_res)

        x31 = x31 + x31_res
        x33 = x33 + x33_res

        x = self.merge_tail(x31, x33)

        return x
