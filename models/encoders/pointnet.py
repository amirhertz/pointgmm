# Code adapted from:
# https://github.com/fxia22/pointnet.pytorch

import torch
import torch.nn as nn
import numpy as np
from models.models_utils import View
from models.encoders.base_encoder import BaseEncoder
from options import Options, RegOptions

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        layers = [
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            torch.nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        ]
        self.conv = nn.Sequential(*layers)
        layers2 = [
            View(-1, 1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 9)
        ]
        self.fc = nn.Sequential(*layers2)

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.conv(x)
        trans = torch.max(trans, 2, keepdim=True)[0]
        trans = self.fc(trans)
        iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)
        iden = iden.to(x.device)
        trans = trans + iden
        trans = trans.view(-1, 3, 3)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        return x, trans


class PointNetfeat(nn.Module):

    def __init__(self, inc):
        super(PointNetfeat, self).__init__()
        if inc == 3:
            # self.stn = STN3d()
            ch_seq = [64, 128, 1024]
        else:
            # self.stn = lambda x: (x, None)
            ch_seq = [max(inc, 256), 512, 1024]
        self.conv1 = torch.nn.Conv1d(inc, ch_seq[0], 1)
        self.conv2 = torch.nn.Conv1d(ch_seq[0], ch_seq[1], 1)
        self.conv3 = torch.nn.Conv1d(ch_seq[1], ch_seq[2], 1)
        self.bn1 = nn.BatchNorm1d(ch_seq[0])
        self.bn2 = nn.BatchNorm1d(ch_seq[1])
        self.bn3 = nn.BatchNorm1d(ch_seq[2])


    def forward(self, x):
        x = x.transpose(2, 1)
        n_pts = x.size()[2]
        # x, trans = self.stn(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class PointNet(BaseEncoder):
    def __init__(self, opt: Options):
        super(PointNet, self).__init__(opt)
        self.feat = PointNetfeat(self.inc)
        layers = [
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, opt.dim_z)
        ]
        self.fc = nn.Sequential(*layers)

    def main(self, x):
        x = self.feat(x)
        x = self.fc(x)
        return x


class PointNetDual(nn.Module):
    def __init__(self, opt: RegOptions):
        super(PointNetDual, self).__init__()
        self.shape_encoder = PointNet(opt)
        self.cat_dim = 3 + opt.dim_z
        self.t_encoder = nn.Sequential(
            PointNetfeat(self.cat_dim),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, opt.dim_t)
        )
        self.split_shape = (3, 2)
        t_extract = [nn.Linear(opt.dim_t, 64), nn.BatchNorm1d(64),nn.ReLU(True)]
        for _ in range(1, opt.trans_layers):
            t_extract += [nn.Linear(64, 64), nn.BatchNorm1d(64),nn.ReLU(True)]
        self.t_extract = nn.Sequential(
            *t_extract,
            nn.Linear(64, sum(self.split_shape))
        )

    def make_transform(self, t_hide):
        raw = self.t_extract(t_hide)
        trnl, angle = torch.split(raw, self.split_shape, dim=1)
        angle = angle / torch.norm(angle, 2, 1)[:, None]
        return trnl, angle

    def forward(self, x):
        z_shape = self.shape_encoder(x)[0]
        z_hide = torch.cat((x, z_shape.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)), 2)
        t_hide = self.t_encoder(z_hide)
        trnl, angle = self.make_transform(t_hide)
        return z_shape, t_hide, trnl, angle
