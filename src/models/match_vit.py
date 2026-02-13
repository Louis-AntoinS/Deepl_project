from torch import nn
from src.models.mobilevit import MV2Block
from src.modules.match_vit_blocks import conv_stem,inverted_residual
from src.models.vit import MaSSA_Block

class MatchVit(nn.Module):
    def __init__(self,num_classes,expansion=2,alpha = 0.75):
        super().__init__()
            
        self.stem = nn.Sequential(
            conv_stem(in_channels=3,out_channels=32*alpha),
            inverted_residual(in_channels=32*alpha,out_channels=32*alpha,expansion_factor=expansion)
        )
        
        self.stage1 = nn.Sequential(
            inverted_residual(in_channels=32*alpha,out_channels=64*alpha,expansion_factor=expansion),
            MaSSA_Block(attn_kwargs={'k': 4, 'gamma':0.5},expansion_factor=expansion)
            
        )
        
        self.stage2 = nn.Sequential(
            inverted_residual(in_channels=64*alpha,out_channels=128*alpha,expansion_factor=expansion),
            MaSSA_Block(attn_kwargs={'k': 8, 'gamma':0.5},expansion_factor=expansion)
            
        )
        
        self.stage3 = nn.Sequential(
            inverted_residual(in_channels=128*alpha,out_channels=256*alpha,expansion_factor=expansion),
            MaSSA_Block(attn_kwargs={'k': 16, 'gamma':0.5},expansion_factor=expansion)
            
        )
        
        self.stage4 = nn.Sequential(
            inverted_residual(in_channels=256*alpha,out_channels=512*alpha,expansion_factor=expansion),
            MaSSA_Block(attn_kwargs={'k': 32, 'gamma':0.5},expansion_factor=expansion)
        )
        
        self.pool = nn.AvgPool2d(4,1)
        self.classifier = nn.Linear(512*alpha,num_classes)
    
    def forward(self,x):
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.pool(x).view(-1, x.shape[1])
        x = self.classifier(x)
        
        return x
    

    