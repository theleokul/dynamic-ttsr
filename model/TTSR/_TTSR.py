import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
import MainNet, MainNet_exp_3_1, LTE, SearchTransfer


class TTSR(nn.Module):
    def __init__(self, n_feats, num_res_blocks, res_scale, experiment='3_1'):
        super(TTSR, self).__init__()
        self.num_res_blocks = list( map(int, num_res_blocks.split('+')) )
        self.experiment = experiment

        if experiment == '3_1':
            print('Running experiment 3_1.')
            self.MainNet = MainNet_exp_3_1.MainNet(
                num_res_blocks=self.num_res_blocks
                , n_feats=n_feats 
                , res_scale=res_scale
            )
        else:
            self.MainNet = MainNet.MainNet(
                num_res_blocks=self.num_res_blocks
                , n_feats=n_feats
                , res_scale=res_scale
            )

        self.LTE      = LTE.LTE(requires_grad=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        _, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)
        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)

        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        return sr, S, T_lv3, T_lv2, T_lv1
