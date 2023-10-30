import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
import random
# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
import numpy as np

from models.memory_module import MemModule 
from models.model_point_cloud import PointNet2SemSegSSG 
from models.model_mem_encoder import Encoder 
from models.model_mem_obj_encoder import ObjEncoder 
from models.model_mem_act_encoder import ActEncoder 
from models.model_mem_decoder import Decoder 

import pdb 


class network(nn.Module):
    def __init__(self, input_dim=16, 
                       pnpp_feat_dim=128, 
                       hidden_feat_dim=128, 
                       feat_dim=128, 
                       hidden_dim = 2, 
                       mem_dim=200):
        super(network, self).__init__()

        self.obj_encoder = ObjEncoder(hidden_dim=hidden_dim, 
                                      input_dim=input_dim, 
                                      pnpp_feat_dim=pnpp_feat_dim, 
                                      hidden_feat_dim=hidden_feat_dim)

        self.act_encoder = ActEncoder(hidden_dim=hidden_dim, 
                                      input_dim=input_dim, 
                                      pnpp_feat_dim=pnpp_feat_dim, 
                                      hidden_feat_dim=hidden_feat_dim)
 
        self.encoder = Encoder(hidden_dim=hidden_dim, 
                                      input_dim=input_dim, 
                                      pnpp_feat_dim=pnpp_feat_dim, 
                                      hidden_feat_dim=hidden_feat_dim)

        self.merge_encoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), 
                    nn.ReLU(), 
                    nn.Linear(hidden_dim, hidden_dim), 
                )
 
        self.decoder = Decoder(feat_dim=pnpp_feat_dim, 
                               out_dim=input_dim)


        self.mem = MemModule(mem_dim=mem_dim, 
                             feat_dim=pnpp_feat_dim, 
                             shrink_thres=1/(mem_dim*2), 
                             )


    def forward(self, dir, dis, push_dis, ctpt, joint_info, 
                          input_pcs, start_pos, end_pos, f_dir, 
                         dropout, dropout_cnt=16, repeat=False):

        batch_size = input_pcs.shape[0]
        idxl = np.array(random.sample([0,2,4,6,8,10,12,14], dropout_cnt))
        idxr = np.array(random.sample([1,3,5,7,9,11,13,15], dropout_cnt))
        # idxr = np.array(random.sample([21,22,23,24,25,26,27,28,29,30], 1))

        if dropout == 'random':
            idx = np.arange(batch_size)
            np.random.shuffle(idx)
            idx = idx[:dropout_cnt]
            mean_hidden_info, act_list, pc_feat =\
              self.encoder(dir[idx, :], 
                    dis[idx], 
                    push_dis[idx], 
                    ctpt[idx, :], 
                    joint_info[idx, :], 
                    input_pcs[idx, :], 
                    start_pos[idx], 
                    end_pos[idx], 
                    f_dir[idx], repeat=repeat)
            obj_emb =\
              self.obj_encoder(\
                    pc_feat, 
                    dir[idx, :], 
                    dis[idx], 
                    push_dis[idx], 
                    ctpt[idx, :], 
                    joint_info[idx, :], 
                    input_pcs[idx, :], 
                    start_pos[idx], 
                    end_pos[idx], 
                    f_dir[idx], repeat=repeat)
            act_emb =\
              self.act_encoder(dir[idx, :], 
                    dis[idx], 
                    push_dis[idx], 
                    ctpt[idx, :], 
                    joint_info[idx, :], 
                    input_pcs[idx, :], 
                    start_pos[idx], 
                    end_pos[idx], 
                    f_dir[idx], repeat=repeat)
 
        else:
            mean_hidden_info, act_list, pc_feat =\
              self.encoder(dir, 
                    dis, 
                    push_dis, 
                    ctpt, 
                    joint_info, 
                    input_pcs, 
                    start_pos, 
                    end_pos, 
                    f_dir, repeat=repeat)
            obj_emb =\
              self.obj_encoder(
                    pc_feat, 
                    dir, 
                    dis, 
                    push_dis, 
                    ctpt, 
                    joint_info, 
                    input_pcs, 
                    start_pos, 
                    end_pos, 
                    f_dir, repeat=repeat)
            act_emb =\
              self.act_encoder(dir, 
                    dis, 
                    push_dis, 
                    ctpt, 
                    joint_info, 
                    input_pcs, 
                    start_pos, 
                    end_pos, 
                    f_dir, repeat=repeat)

        interact_enc_query = self.merge_encoder(mean_hidden_info)
        mem_out = self.mem(interact_enc_query)
        act_enc_query = self.merge_encoder(act_emb)
        act_mem_out = self.mem(act_enc_query)
        obj_enc_query = self.merge_encoder(obj_emb)
        obj_mem_out = self.mem(obj_enc_query)

        mem_feat = mem_out['output']
        mem_att = mem_out['att'] 
        act_mem_feat = act_mem_out['output'] 
        act_mem_att = act_mem_out['att']
        obj_mem_feat = obj_mem_out['output'] 
        obj_mem_att = obj_mem_out['att'] 

        rec_list = self.decoder(mem_feat, obj_emb) 
        act_rec_list = self.decoder(act_mem_feat, obj_emb) 
        obj_rec_list = self.decoder(obj_mem_feat, obj_emb) 

        return_dict = {'rec_list': rec_list, 
                       'act_rec_list': act_rec_list, 
                       'obj_rec_list': obj_rec_list, 
                       'att': mem_att, 
                       'act_list': act_list, 
                       'act_att': act_mem_att, 
                       'obj_att': obj_mem_att, 
                       } 
        #return_dict.update(reg_dict) 
        return return_dict 

