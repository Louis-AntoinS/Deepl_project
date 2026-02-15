from ast import mod
from math import exp
from sympy import false
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


class Inverted_residual(nn.Module):
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
                nn.SiLU(inplace=True),
                # depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride, padding=1, groups=hidden_dim, bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
                # projection layer 
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias = False),
                nn.BatchNorm2d(out_channels)
            )
                 
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
                 

class Conv_stem(nn.Module):
     def __init__(self, in_channels, out_channels) :
          super().__init__()

          self.conv_3x3 = nn.Sequential(
               nn.Conv2d(in_channels,out_channels,kernel_size=3, stride = 2, padding = 1, bias= False),
               nn.BatchNorm2d(out_channels),
               nn.SiLU(inplace=True)
          )

          self.mv2_block = Inverted_residual(out_channels, out_channels,stride = 2, expansion_factor=2)

     def forward(self, x):
        # x : [Batch, 3, 224, 224]
        x = self.conv_3x3(x)
        # x : [Batch, 24, 112, 112]
        out = self.mv2_block(x)
        # out : [Batch, 24, 56, 56]
        return out
         

class MaViTLayer(nn.Module):
    def __init__(self, channels, k,attentionMaSSA = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.attentionMaSSA = attentionMaSSA
        
        
        if attentionMaSSA:
            self.attention = MaSSA(embed_dim=channels, k=k)
            self.context_gated_ffn = ContextGatedFFN(embed_dim=channels, k=k, gamma=0.5, expansion_factor=2)
        else:
            self.attention = MHSA(embed_dim=channels, num_heads=k)
            self.context_gated_ffn = MLP(embed_dim=channels, expansion_factor=2)
        
    def forward(self, x):
        
        if self.attentionMaSSA:
            attn_out, context_vectors = self.attention(self.ln1(x))
            
        else:
            attn_out = self.attention(self.ln1(x))
            
        x = x + attn_out
        
        if self.attentionMaSSA:
            x = x + self.context_gated_ffn(self.ln2(x), context_vectors)
        else:
            x = x + self.context_gated_ffn(self.ln2(x))
            
        return x
    

class MatchViTBlock(nn.Module):
    def __init__(self, channels, num_heads, attentionMaSSA = True):
        super().__init__()
        
        # Local representation
        self.local_rep = nn.Sequential(
              nn.Conv2d(in_channels=channels,out_channels=channels, kernel_size=3 ,padding=1, groups= channels, bias = False),
              nn.BatchNorm2d(channels),
              nn.SiLU(),
              nn.Conv2d(channels, channels, 1, bias=False),
              nn.BatchNorm2d(channels),
              nn.SiLU(inplace=True)
         )
        
        # Global representation 
        self.transformer = MaViTLayer(channels, num_heads, attentionMaSSA)

    def forward(self, x):
        # x : [Batch,C,H,W]

        # Local representations
        x = x + self.local_rep(x)

        # Global representations
    
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1,2) # x : [Batch,N,C]        
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, C, H, W) # x : [Batch,C,H,W]
        return x 



        


        


                 








