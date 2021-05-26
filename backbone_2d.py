import numba
import torch
import numpy as np
from torch import nn
from math import sqrt
import torchvision
import torch.nn.functional as F
from scipy import stats

from utils import *

class Backbone2D(nn.Module):
    def __init__(self,
                channels_for_block = [128, 256, 512],
                upsample_strides = [1, 2, 4],
                conv_size_for_block = [4, 6, 6],
                n_cfg = {'out_channel_size': 128, 'classes': 4}):
        """
        Backbone2D class is get feature map for head detection (only ssd head detection).
        :param n_cfg[classes]: class size (default 4: car, ciclyst, pedastrian, background)
        :param n_cfg[out_channel_size]: out channel size (default: 128)
        :param channels_for_block: array for construct of 2d convolution block (default: 128, 256, 512)
        :param upsample_strides: (default: 1, 2, 4)
        :param conv_size_for_block: array for construct of 2d convolution block (default: 4, 6, 6)
        """
        super().__init__()

        # Concatenate all feature_map (view backbone architecture https://arxiv.org/pdf/1812.05784.pdf) 
        assert len(channels_for_block) == 3

        super(nn.PredictionConvolutions, self).__init__()

        self.cfg = n_cfg
        boxes = self.cfg['boxes']
        out_channel_size = self.cfg['out_channel_size']

        self.dw_block = []
        self.ups_block = []

        # Downsampling
        for i in range(channels_for_block):
            conv_sequential = [nn.Conv2d(channels_for_block[i] / 2, channels_for_block[i], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                               nn.BatchNorm2d(channels_for_block[i], eps=1e-03, momentum=0.1, affine=True, track_running_stats=True),
                               nn.ReLU(inplace=True)]
            for j in conv_size_for_block[i]:
                conv_sequential.extend([nn.Conv2d(channels_for_block[i], channels_for_block[i], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                        nn.BatchNorm2d(channels_for_block[i], eps=1e-03, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True)])
            self.dws_block.append(nn.Sequential(*conv_sequential))
       
        # Upsampling
        for i in range(channels_for_block):
            stride = upsample_strides[i] if upsample_strides[i] >= 1 else np.round(1 / upsample_strides[i]).astype(np.int)
            self.ups_block.append(nn.Sequential(nn.ConvTranspose2d(channels_for_block[i], out_channel_size, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False),
                                                nn.BatchNorm2d(out_channel_size, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True),
                                                nn.ReLU(inplace=True)))

        self.num_bev_features = out_channel_size * len(channels_for_block)

    def forward(self, feature_map):
        # batch_size = feature_map.size(0)

        feature_map_buf = self.dws_block[0](feature_map)
        ups = [ self.ups_block[0](feature_map_buf) ]

        for i in range(1, len(self.dws_block)):
            feature_map_buf = self.dws_block[i](feature_map_buf)
            out = self.ups_block[i](feature_map_buf)
            ups.append(out)

        # Concatenate all feature_map (view backbone architecture https://arxiv.org/pdf/1812.05784.pdf) 
        out_feature_map = torch.cat(ups, dim=1)
        return out_feature_map
