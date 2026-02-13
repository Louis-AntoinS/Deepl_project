from ast import mod
from math import exp
from typing import Self
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.utils as vutils
import torchvision
from PIL import Image
import torch.nn.functional as F
import numpy as np
from src.modules.attention import MHSA, MaSSA
from src.modules.ffn import MLP, ContextGatedFFN


class inverted_residual(nn.Module):
    def __init__(self,in_channels, out_channels, stride = 2 , expansion_factor=2):
            super().__init__()
            
            self.stride = stride
            self.in_channels = in_channels
            self.out_channels = out_channels
            hidden_dim = int(in_channels * expansion_factor)
            self.use_res_connect = self.stride == 1 and in_channels == out_channels

            self.conv = nn.Sequential(
                # expansion layer
                nn.Conv2d(self.in_channels,hidden_dim,kernel_size=1,stride=1,padding = 0, bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride, padding=1, groups=hidden_dim, bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # projection layer 
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias = False),
                nn.BatchNorm2d(out_channels)
            )
                 
            def forward(self, x):
                if self.use_res_connect:
                    return x + self.conv(x)
                else:
                    return self.conv(x)
                 
                 
class conv_stem(nn.Module):
     def __init__(self, in_channels, out_channels) :
          super().__init__()

          self.conv_3x3 = nn.Sequential(
               nn.Conv2d(in_channels,out_channels,kernel_size=3, stride = 2, padding = 1, bias= False),
               nn.BatchNorm2d(out_channels),
               nn.ReLU6(inplace=True)
          )

          self.mv2_block = inverted_residual(out_channels, out_channels,stride = 2, expansion_factor=2)

     def forward(self, x):
        # x : [Batch, 3, 224, 224]
        x = self.conv_3x3(x)
        # x : [Batch, 24, 112, 112]
        out = self.mv2_block(x)
        # out : [Batch, 24, 56, 56]
        return out
     

                 








