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

 
class ActEncoder(nn.Module):
    def __init__(self, hidden_dim=2, input_dim=16, pnpp_feat_dim=128, hidden_feat_dim=128):
        super(ActEncoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.dir_layer = nn.Sequential(
            nn.Linear(3, 16),
        )            

        self.dis_layer = nn.Sequential(
            nn.Linear(1, 16),
        )            

        self.push_dis_layer = nn.Sequential(
            nn.Linear(1, 16),
        )            

        self.ctpt_layer = nn.Sequential(
            nn.Linear(3, 16),
        )            

        self.joint_info_layer = nn.Sequential(
            nn.Linear(4, 16),
        )            

        self.start_pos_layer = nn.Sequential(
            nn.Linear(1, 16),
        )            

        self.end_pos_layer = nn.Sequential(
            nn.Linear(1, 16),
        )            

        self.f_dir_layer = nn.Sequential(
            nn.Linear(3, 16),
        )            


        self.cat_layer = nn.Sequential(
                nn.Linear(8 * 16, hidden_feat_dim), 
                nn.ReLU(),
                nn.Linear(hidden_feat_dim, hidden_feat_dim),
                )

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_feat_dim),
            nn.ReLU(),
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
        )

        self.gather_layer = nn.Sequential(
            nn.Linear(hidden_feat_dim * 3, hidden_feat_dim),
            nn.ReLU(),
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
        ) 

        self.hidden_info_encoder = nn.Linear(hidden_feat_dim, hidden_dim)

    def forward(self, dir, dis, push_dis, 
                      ctpt, joint_info, pcs, 
                      start_pos, end_pos, f_dir, repeat=False):

        batch_size = dir.shape[0]
        dir = dir.view(batch_size, -1)
        f_dir = f_dir.view(batch_size, -1)
        dis = dis.view(batch_size, -1)
        ctpt = ctpt.view(batch_size, -1)
        start_pos = start_pos.view(batch_size, -1)
        end_pos = end_pos.view(batch_size, -1)
        joint_info = joint_info.view(batch_size, -1)

        dir_emb = self.dir_layer(dir) 
        dis_emb = self.dis_layer(dis) 
        push_dis_emb = self.push_dis_layer(push_dis)
        ctpt_emb = self.ctpt_layer(ctpt) 
        joint_info_emb = self.joint_info_layer(joint_info)
        start_pos_emb = self.start_pos_layer(start_pos)
        end_pos_emb = self.end_pos_layer(end_pos) 
        f_dir_emb = self.f_dir_layer(f_dir) 

        emb_cat = torch.cat([dir_emb, dis_emb, push_dis_emb, ctpt_emb, joint_info_emb, start_pos_emb, end_pos_emb, f_dir_emb], dim=-1)
        emb_out = self.cat_layer(emb_cat) 

        x = torch.cat([dir, dis, push_dis, ctpt, joint_info, start_pos, end_pos, f_dir], dim=-1)
        hidden_feat = self.mlp1(x)
        merge_feat = hidden_feat + emb_out 

        feat_out = self.hidden_info_encoder(merge_feat)

        return feat_out


 
