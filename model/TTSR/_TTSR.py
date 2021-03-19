import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

DIR_PATH = pathlib.Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH))
import MainNet, MainNet_exp_3_1, MainNet_exp_3_2, LTE, SearchTransfer


class TTSR(nn.Module):
    def __init__(
        self
        , n_feats
        , num_res_blocks
        , res_scale
        , LTE_requires_grads=[True, True, True]
        , experiment=''
    ):

        super(TTSR, self).__init__()
        self.num_res_blocks = list( map(int, num_res_blocks.split('+')) )
        self.experiment = experiment

        if experiment == '3_1':
            print('Running experiment 3_1...')
            self.MainNet = MainNet_exp_3_1.MainNet(
                num_res_blocks=self.num_res_blocks
                , n_feats=n_feats 
                , res_scale=res_scale
            )
        elif experiment == '3_2':
            print('Running experiment 3_2...')
            self.MainNet = MainNet_exp_3_2.MainNet(
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

        self.LTE      = LTE.LTE(requires_grads=LTE_requires_grads)
        self.LTE_copy = LTE.LTE(requires_grads=[False, False, False]) ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            
            if self.experiment == '3_1':
                sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
                # output = sr_lv1, None, sr_lv3
                output = sr_lv1, sr_lv2, sr_lv3
            elif self.experiment == '3_2':
                _, _, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
                output = None, None, sr_lv3
            else:
                sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
                output = sr_lv1, sr_lv2, sr_lv3

            return output

        _, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)
        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)

        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        if self.experiment == '3_1':
            # output = sr, S, T_lv3, None, T_lv1
            output = sr, S, T_lv3, T_lv2, T_lv1
        elif self.experiment == '3_2':
            output = sr, S, T_lv3, None, None
        else:
            output = sr, S, T_lv3, T_lv2, T_lv1

        return output
