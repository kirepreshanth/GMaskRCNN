##########################################################################################################################
# This code was refactored out of the generative-inpainting-pytorch/model/networks.py file
#
# Ang, D (2019) generative-inpainting-pytorch [sourcecode]
# https://github.com/DAA233/generative-inpainting-pytorch/blob/master/model/networks.py
# (fork backup) https://github.com/kirepreshanth/generative-inpainting-pytorch/blob/master/model/networks.py
#
# This was done so that variations of the Generators could be used
# depending on the architecture of the config file.
##########################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from model.helper import dis_conv
from model.default import CoarseGenerator, FineGenerator
from model.unet import CoarseUNetGenerator, FineUNetGenerator
from model.fully_conv_network import CoarseFCN8Generator
from model.segnet import CoarseSegNetGenerator, FineSegNetGenerator


class Generator(nn.Module):
    def __init__(self, config, use_cuda, device_ids):
        super(Generator, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']
        self.architecture = config['architecture']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        if self.architecture == 'default':
            self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
            self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
        elif self.architecture == 'unet':
            self.coarse_generator = CoarseUNetGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
            self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
        elif self.architecture == 'fcn8':
            self.coarse_generator = CoarseFCN8Generator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
            self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
        elif self.architecture == 'segnet':
            self.coarse_generator = CoarseSegNetGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
            self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
        

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow
    
class LocalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x

class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)       

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x