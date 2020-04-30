#######################################################################################################################
# Copyright(c) 2019 - present, Mahmoud Afifi
# York University, Canada
# Email: mafifi @ eecs.yorku.ca - m.3afifi @ gmail.com
#######################################################################################################################
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
# All rights reserved.
#
# Please cite the following work if this program is used:
# 1 - Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S.Brown.When Color Constancy Goes Wrong: Correcting
# Improperly White-Balanced Images. In CVPR, 2019
# 2 - Mahmoud Afifi and Michael S.Brown.Sensor Independent Illumination Estimation for DNN Models. In BMVC, 2019
#######################################################################################################################
# Create a histogram RGB - uv block to plugin it into a CNN model.There are two options: 1) a histogram block with two
# learnable parameters that control the histogram generation process and 2) a histogram block without learnable
# parameters.

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class RGBuvHistBlock(nn.Module):
    def __init__(self, h= 61, insz= 150, C= [1, 1, 1], sigmaU= [0.05, 0.05, 0.05], sigmaV= [0.05, 0.05, 0.05], learnable= 0):
        super(RGBuvHistBlock, self).__init__()
        assert C.__len__() is 3, 'C should be a 3D vector'
        assert sigmaU.__len__() is 3, 'Sigma U should be a 3D vector'
        assert sigmaV.__len__() is 3, 'Sigma V should be a 3D vector'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.h = h
        self.insz = insz
        self.learnable = learnable
        self.eps = 2.2204e-16
        self.eps_ = 6.4 / self.h
        self.A = torch.tensor(np.linspace(-3.2, 3.0951, num=self.h)).to(self.device)
        if self.learnable is True:
            self.C = torch.tensor(C, requires_grad=True).to(self.device).requires_grad
            self.sigmaU = torch.tensor(sigmaU, requires_grad=True).to(self.device).requires_grad
            self.sigmaV = torch.tensor(sigmaV, requires_grad=True).to(self.device).requires_grad
        else:
            self.C = torch.tensor(C).to(self.device)
            self.sigmaU = torch.tensor(sigmaU).to(self.device)
            self.sigmaV = torch.tensor(sigmaV).to(self.device)

    def forward(self, X):
        if X.size(2) != self.insz or X.size(3) != self.insz:
            X = F.interpolate(X, size=(self.insz, self.insz), mode='nearest')
        hists = torch.tensor(np.zeros((X.size(0), 3, self.h, self.h), dtype=np.float32)).to(self.device)
        L = X.size(0)  # size of mini-batch
        #print("size is %d" % L)
        X = torch.unbind(X, dim=0)
        for l in range(L):
            I = torch.transpose(torch.reshape(X[l], (3, -1)), 0, 1)
            Iy = torch.sqrt(torch.pow(I[:, 0], 2) + torch.pow(I[:, 1], 2) + torch.pow(I[:, 2], 2))
            for i in range(3):
                r = list(set([0, 1, 2]) - set([i]))
                Iu = torch.log(torch.abs(I[:, i]) + self.eps) - torch.log(torch.abs(I[:, r[0]]) + self.eps)
                Iv = torch.log(torch.abs(I[:, i]) + self.eps) - torch.log(torch.abs(I[:, r[1]]) + self.eps)
                diff_u = abs((Iu.repeat((1, self.A.size(0))).view((Iu.size(0), self.A.size(0)))).to(self.device) -
                             (self.A.repeat((Iu.size(0), 1)).view((Iu.size(0), self.A.size(0)))).to(self.device))
                diff_v = abs((Iv.repeat((1, self.A.size(0))).view((Iv.size(0), self.A.size(0)))).to(self.device) -
                             (self.A.repeat((Iv.size(0), 1)).view((Iv.size(0), self.A.size(0)))).to(self.device))
                if self.learnable == 1:
                    diff_u = torch.exp(-(torch.reshape((torch.pow(torch.reshape(diff_u, (-1, 1)), 2)),
                                                   (-1, self.A.size(0))) / (torch.pow(self.sigmaU[i], 2) + self.eps).repeat((diff_u.size(0), self.h))))
                    diff_v = torch.exp(-(torch.reshape((torch.pow(torch.reshape(diff_v, (-1, 1)), 2)),
                                                   (-1, self.A.size(0))) / (torch.pow(self.sigmaV[i], 2) + self.eps).repeat((diff_v.size(0), self.h))))
                else:
                    diff_u = (torch.reshape((torch.reshape(diff_u, (-1, 1)) <= self.eps_ / 2), (-1, self.A.size(0))))
                    diff_v = (torch.reshape((torch.reshape(diff_v, (-1, 1)) <= self.eps_ / 2), (-1, self.A.size(0))))

                diff_u = diff_u.type(torch.float32)
                diff_v = diff_v.type(torch.float32)

                hists[l, i, :, :] = (torch.mm(torch.t(torch.mul((
                    Iy.repeat((1, self.A.size(0))).view((Iy.size(0), self.A.size(0))).to(self.device)), diff_u)), diff_v))

                if self.learnable == 1:
                    hists[l, i, :, :] = torch.mul(torch.sqrt(hists[l, i, :, :]), self.C[i].repeat(self.h, self.h))
                else:
                    hists[l, i, :, :] = torch.mul(torch.sqrt(hists[l, i, :, :]), (self.C[i] / I.size(0)).repeat(self.h, self.h))
        return hists

    def set_C(self, C):
        if self.learnable:
            self.C = torch.squeeze(torch.tensor(C, requires_grad=True)).to(self.device)
        else:
            self.C = torch.squeeze(torch.tensor(C, requires_grad=False)).to(self.device)

    def set_sigmaU(self, sigmaU):
        if self.learnable:
            self.sigmaU = torch.squeeze(torch.tensor(sigmaU, requires_grad=True)).to(self.device)
        else:
            self.sigmaU = torch.squeeze(torch.tensor(sigmaU, requires_grad=False)).to(self.device)

    def set_sigmaV(self, sigmaV):
        if self.learnable:
            self.sigmaV = torch.squeeze(torch.tensor(sigmaV, requires_grad=True)).to(self.device)
        else:
            self.sigmaV = torch.squeeze(torch.tensor(sigmaV, requires_grad=False)).to(self.device)


