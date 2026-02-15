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



class MatchVitStage(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, num_heads,reps, attentionMaSSA = True):
        super().__init__()
        self.inverted_residual = Inverted_residual(in_channels, out_channels, expansion_factor)
        
        mavit_blocks = []
        for _ in range(reps):
            mavit_blocks.append(MatchViTBlock(channels=out_channels, num_heads=num_heads, attentionMaSSA=attentionMaSSA))
        self.mavit_blocks = nn.Sequential(*mavit_blocks)
        
    def forward(self, x):
        x = self.inverted_residual(x)
        x = self.mavit_blocks(x)
        return x


class MatchVit(nn.Module):
    def __init__(self,num_classes,expansion=2,alpha = 0.75, attentionMaSSA = True):
        super().__init__()
            
        self.stem = Conv_stem(in_channels=3, out_channels=int(32*alpha))
        
        self.stage1 = nn.Sequential(
            MatchVitStage(in_channels=int(32*alpha), out_channels=int(64*alpha), expansion_factor=expansion, num_heads=4, reps=2, attentionMaSSA=attentionMaSSA),
        )
        
        self.stage2 = nn.Sequential(
            MatchVitStage(in_channels=int(64*alpha), out_channels=int(128*alpha), expansion_factor=expansion, num_heads=8, reps=3, attentionMaSSA=attentionMaSSA),
        )
        
        self.stage3 = nn.Sequential(
            MatchVitStage(in_channels=int(128*alpha), out_channels=int(256*alpha), expansion_factor=expansion, num_heads=16, reps=4, attentionMaSSA=attentionMaSSA),
        )
        
        self.stage4 = nn.Sequential(
            MatchVitStage(in_channels=int(256*alpha), out_channels=int(512*alpha), expansion_factor=expansion, num_heads=32, reps=1, attentionMaSSA=attentionMaSSA)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(int(512*alpha),num_classes)
        self.droupout = nn.Dropout(0.3)
    def forward(self,x):
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.pool(x).view(x.shape[0], -1)
        x = self.droupout(x)
        x = self.classifier(x)
        
        return x
    

if __name__ == "__main__":
    model = MatchVit(num_classes=1000, expansion=2, alpha=1)
    x = torch.randn(1,3,224,224)
    out = model(x)
    print(out.shape)
    print(count_parameters(model))

