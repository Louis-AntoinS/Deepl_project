from itertools import count
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
from src.models.mobilevit import MV2Block
from src.modules.match_vit_blocks import Conv_stem,Inverted_residual, MatchViTBlock
from src.models.vit import MaSSA_Block
from src.utils import count_parameters



class MatchVit(nn.Module):
    def __init__(self,num_classes,expansion=2,alpha = 0.75):
        super().__init__()
            
        self.stem = Conv_stem(in_channels=3, out_channels=int(32*alpha))
        
        self.stage1 = nn.Sequential(
            Inverted_residual(in_channels=int(32*alpha),out_channels=int(64*alpha),expansion_factor=expansion),
            MatchViTBlock(channels = int(64*alpha), num_heads=4)
        )
        
        self.stage2 = nn.Sequential(
            Inverted_residual(in_channels=int(64*alpha),out_channels=int(128*alpha),expansion_factor=expansion),
            MatchViTBlock(channels = int(128*alpha), num_heads=8)

        )
        
        self.stage3 = nn.Sequential(
            Inverted_residual(in_channels=int(128*alpha),out_channels=int(256*alpha),expansion_factor=expansion),
            MatchViTBlock(channels = int(256*alpha), num_heads=16)
            
        )
        
        self.stage4 = nn.Sequential(
            Inverted_residual(in_channels=int(256*alpha),out_channels=int(512*alpha),expansion_factor=expansion),
            MatchViTBlock(channels = int(512*alpha), num_heads=32)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(int(512*alpha),num_classes)
    
    def forward(self,x):
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.pool(x).view(x.shape[0], -1)
        x = self.classifier(x)
        
        return x
    

if __name__ == "__main__":
    model = MatchVit(num_classes=1000, expansion=2, alpha=0.75)
    x = torch.randn(1,3,224,224)
    out = model(x)
    print(out.shape)
    print(count_parameters(model))

