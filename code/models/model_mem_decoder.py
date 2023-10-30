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
 
class Decoder(nn.Module): 
    def __init__(self, feat_dim, out_dim): 
        super(Decoder, self).__init__() 
        self.emb = nn.Sequential(
                nn.Linear(feat_dim, feat_dim), 
                #nn.LayerNorm((feat_dim, ), eps=1e-5), 
                nn.ReLU(), 
                nn.Linear(feat_dim, feat_dim), 
                #nn.LayerNorm((feat_dim, ), eps=1e-5), 
                nn.ReLU(), 
                nn.Linear(feat_dim, feat_dim), 
                #nn.LayerNorm((feat_dim, ), eps=1e-5), 
                )
        self.pc_dec = nn.Sequential(
                nn.Linear(feat_dim, feat_dim), 
                nn.ReLU(), 
                nn.Linear(feat_dim, feat_dim), 
                )   

        self.cat_layer = nn.Sequential(
            nn.BatchNorm1d(feat_dim * 2), 
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(), 
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(), 
            nn.Linear(feat_dim, feat_dim), 
        )

        self.dir_reg = nn.Sequential(
            nn.Linear(feat_dim, 3),
        )            

        self.dis_reg = nn.Sequential(
            nn.Linear(feat_dim, 1),
        )            

        self.push_dis_reg = nn.Sequential(
            nn.Linear(feat_dim, 1),
        )            

        self.ctpt_reg = nn.Sequential(
            nn.Linear(feat_dim, 3),
        )            

        self.joint_info_reg = nn.Sequential(
            nn.Linear(feat_dim, 4),
        )            

        self.start_pos_reg = nn.Sequential(
            nn.Linear(feat_dim, 1),
        )            

        self.end_pos_reg = nn.Sequential(
            nn.Linear(feat_dim, 1),
        )            

        self.f_dir_reg = nn.Sequential(
            nn.Linear(feat_dim, 3),
        )            
 

    def forward(self, feat, pc_feat): 
        rec_emb = self.emb(feat) 
        pc_emb = self.pc_dec(pc_feat) 

        cat_emb = torch.cat([rec_emb, pc_emb], dim=-1)
        act_emb = self.cat_layer(cat_emb) 

        dir_out = self.dir_reg(act_emb)
        dis_out = self.dis_reg(act_emb) 
        push_dis_out = self.push_dis_reg(act_emb) 
        ctpt_out = self.ctpt_reg(act_emb) 
        joint_info_out = self.joint_info_reg(act_emb) 
        start_pos_out = self.start_pos_reg(act_emb) 
        end_pos_out = self.end_pos_reg(act_emb) 
        f_dir_out = self.f_dir_reg(act_emb) 
        
        return [dir_out, dis_out, push_dis_out, ctpt_out, \
               joint_info_out, start_pos_out, end_pos_out, f_dir_out]  
 
