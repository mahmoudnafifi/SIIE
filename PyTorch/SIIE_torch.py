#######################################################################################################################
# Copyright(c) 2019 - present, Mahmoud Afifi
# York University, Canada
# Email: mafifi @ eecs.yorku.ca - m.3afifi @ gmail.com
#######################################################################################################################
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
# All rights reserved.
#
# Please cite the following work if this program is used:
# Mahmoud Afifi and Michael S.Brown.Sensor Independent Illumination Estimation for DNN Models. In BMVC, 2019
#######################################################################################################################


import torch
import torch.nn.functional as F
import torch.nn as nn
import RGBuvHistBlock as histBlock
from torchvision import transforms
from random import random
from six.moves import cPickle as pickle #for performance


class SIIE_torch(nn.Module):
    def __init__(self):
        super(SIIE_torch, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.hist_sensor = histBlock.RGBuvHistBlock(h = 61, insz = 150,
                                                    sigmaU = [random() * 0.02, random() * 0.02, random() * 0.02],
                                                    sigmaV = [random() * 0.02, random() * 0.02, random() * 0.02],
                                                    C = [1, 1, 1], learnable= 1)
        self.sensor_conv1 = nn.Conv2d(3, 128, 5, stride=2, padding=0).to(device=self.device)
        self.sensor_conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=0).to(device=self.device)
        self.sensor_conv3 = nn.Conv2d(256, 512, 2, stride=1, padding=0).to(device=self.device)
        self.sensor_fc = nn.Linear(512 * 13 * 13, 9, bias=False).to(device=self.device)  # 13*13 from hist dimension

        self.hist_illum = histBlock.RGBuvHistBlock(h=61, insz=150,
                                                    sigmaU=[random() * 0.02, random() * 0.02, random() * 0.02],
                                                    sigmaV=[random() * 0.02, random() * 0.02, random() * 0.02],
                                                    C = [1, 1, 1], learnable=1)
        self.illum_conv1 = nn.Conv2d(3, 128, 5, stride=2, padding=0).to(device=self.device)
        self.illum_conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=0).to(device=self.device)
        self.illum_conv3 = nn.Conv2d(256, 512, 2, stride=1, padding=0).to(device=self.device)
        self.illum_fc = nn.Linear(512 * 13 * 13, 3, bias=False).to(device=self.device)  # 13*13 from hist dimension


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



    def loadModel(self, modelName):
        self.load_state_dict(torch.load(modelName + '.pt'))
        with open(modelName + '_hist.pkl', 'rb') as f:
            hist_data = pickle.load(f)
            self.hist_sensor.set_C(hist_data['hist_sensor_C'])
            self.hist_sensor.set_sigmaU(hist_data['hist_sensor_sigmaU'])
            self.hist_sensor.set_sigmaV(hist_data['hist_sensor_sigmaV'])
            self.hist_illum.set_C(hist_data['hist_ill_C'])
            self.hist_illum.set_sigmaU(hist_data['hist_ill_sigmaU'])
            self.hist_illum.set_sigmaV(hist_data['hist_ill_sigmaV'])
        self.eval()
        return self

    def predict(self, model, image):
        image = transforms.ToTensor()(image).unsqueeze_(0).to(device=self.device)
        hist_input = self.hist_sensor.forward(image)
        latent = F.relu(self.sensor_conv1(hist_input))
        latent = F.relu(self.sensor_conv2(latent))
        latent = F.relu(self.sensor_conv3(latent))
        #latent = latent.view(-1, self.num_flat_features(latent))
        latent = latent.reshape((1, -1))
        sensor_matrix = torch.reshape(torch.abs(self.sensor_fc(latent)), (3, 3)).t()
        n = torch.max(torch.norm(sensor_matrix, 1, -1)) + 0.0001
        sensor_matrix = sensor_matrix / n
        print(sensor_matrix.float())
        mapped_image = torch.mm(torch.reshape(torch.squeeze(image).permute(1, 2, 0), (-1, 3)), sensor_matrix.t())
        mapped_image = torch.reshape(mapped_image, (image.size(2), image.size(3), image.size(1)))
        mapped_image_cpu = mapped_image.cpu().detach().clone().numpy()
        mapped_image = mapped_image.permute(2, 0, 1).unsqueeze(0)
        hist_mapped = self.hist_illum.forward(mapped_image)
        latent = F.relu(self.illum_conv1(hist_mapped))
        latent = F.relu(self.illum_conv2(latent))
        latent = F.relu(self.illum_conv3(latent))
        #latent = latent.view(-1, self.num_flat_features(latent))
        latent = latent.reshape((1, -1))
        est_ill = torch.reshape(torch.abs(self.illum_fc(latent)), (3, 1))

        if torch.det(sensor_matrix).float() == 0.0:
            sensor_matrix = sensor_matrix + torch.rand((3, 3)) / 1000

        est_ill = torch.squeeze(torch.mm(torch.inverse(sensor_matrix), est_ill))


        return (est_ill.cpu()).detach().numpy(), mapped_image_cpu



