#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
import random

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from models.model_point_cloud import PointNet2SemSegSSG 

import numpy as np
 
import pdb 

 
class Encoder(nn.Module):
    def __init__(self, hidden_dim=2, input_dim=16, pnpp_feat_dim=128, hidden_feat_dim=128):
        super(Encoder, self).__init__()

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

        self.pc_emb_layer = nn.Sequential( 
                nn.Linear(pnpp_feat_dim, hidden_feat_dim), 
                nn.ReLU(),
                nn.Linear(hidden_feat_dim, hidden_feat_dim),
                )

        self.cat_layer = nn.Sequential(
                nn.BatchNorm1d(8 * 16), 
                nn.Linear(8 * 16, hidden_feat_dim), 
                nn.ReLU(),
                nn.Linear(hidden_feat_dim, hidden_feat_dim),
                )

        self.act_cat_layer = nn.Sequential(
                nn.BatchNorm1d(6 * 16), 
                nn.Linear(6 * 16, hidden_feat_dim), 
                nn.ReLU(),
                nn.Linear(hidden_feat_dim, hidden_feat_dim),
                )

        self.obj_cat_layer = nn.Sequential(
                nn.BatchNorm1d(2 * 16 + 128), 
                nn.Linear(2 * 16 + 128, hidden_feat_dim), 
                nn.ReLU(),
                nn.Linear(hidden_feat_dim, hidden_feat_dim),
                )

        self.down_emb = nn.Sequential(
                nn.Linear(hidden_feat_dim, hidden_feat_dim // 4), 
                nn.ReLU(), 
                nn.Linear(hidden_feat_dim // 4, hidden_feat_dim // 32), 
                nn.ReLU(), 
                )      
        self.merge_emb = nn.Sequential(
                nn.Linear(hidden_feat_dim * hidden_feat_dim // 32, hidden_feat_dim), 
                nn.ReLU(), 
                nn.Linear(hidden_feat_dim, hidden_feat_dim), 
                )      
 
 
        self.mlp1 = nn.Sequential(
            nn.BatchNorm1d(input_dim), 
            nn.Linear(input_dim, hidden_feat_dim),
            nn.ReLU(),
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
        )

        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(hidden_feat_dim), 
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
        )

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': pnpp_feat_dim})

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
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        pc_feat_ = whole_feats[:, :, 0]

        if repeat:
            pc_feat = pc_feat_.repeat(batch_size, 1)  
        else: 
            pc_feat = pc_feat_ 

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
        act_emb_cat = torch.cat([dir_emb, dis_emb, push_dis_emb, joint_info_emb, start_pos_emb, end_pos_emb], dim=-1)
        act_feat = self.act_cat_layer(act_emb_cat) 
        obj_emb_cat = torch.cat([ ctpt_emb, joint_info_emb, pc_feat ], dim=-1)
        obj_feat = self.obj_cat_layer(obj_emb_cat)

        mul_emb = torch.matmul(act_feat.unsqueeze(2), obj_feat.unsqueeze(1))
        down_emb = self.down_emb(mul_emb) 
        batch_size = obj_feat.size(0) 
        reshape_emb = down_emb.view(batch_size, -1).contiguous()  
        emb_out = self.merge_emb(reshape_emb)

        x = torch.cat([dir, dis, push_dis, ctpt, joint_info, start_pos, end_pos, f_dir], dim=-1)
        act_list = [dir, dis, push_dis, ctpt, joint_info, start_pos, end_pos, f_dir]
        hidden_feat = self.mlp1(x)
        hidden_feat = hidden_feat + emb_out 

        pc_emb_feat = self.pc_emb_layer(pc_feat) 
        hidden_feat = hidden_feat + pc_emb_feat
        out_feat = self.mlp2(hidden_feat)

        return out_feat, act_list, pc_feat     
