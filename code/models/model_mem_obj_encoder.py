#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
import random
# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
import numpy as np
 
import pdb 

 
class ObjEncoder(nn.Module):
    def __init__(self, hidden_dim=2, input_dim=16, pnpp_feat_dim=128, hidden_feat_dim=128):
        super(ObjEncoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.ctpt_layer = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(), 
            nn.Linear(16, 16), 
        )            

        self.mlp1 = nn.Sequential(
            nn.BatchNorm1d(16 + pnpp_feat_dim), 
            nn.Linear(16 + pnpp_feat_dim, hidden_feat_dim),
            nn.ReLU(),
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
            nn.ReLU(), 
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
        )
        self.hidden_info_encoder = nn.Linear(hidden_feat_dim, hidden_dim)


    def forward(self, pc_feat, dir, dis, push_dis, 
                      ctpt, joint_info, pcs, 
                      start_pos, end_pos, f_dir, repeat=False):

        batch_size = dir.shape[0]
        ctpt = ctpt.view(batch_size, -1)
        ctpt_emb = self.ctpt_layer(ctpt) 
        est_inputs = torch.cat([ctpt_emb, pc_feat], dim=-1)

        feat_1 = self.mlp1(est_inputs)
        feat_2 = self.mlp2(feat_1)
        merge_feat = feat_1 + feat_2 
        out_feat = self.hidden_info_encoder(merge_feat)

        return out_feat      


