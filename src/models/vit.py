from ast import mod
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
from src.modules.layers import MLP, PatchEmbed, get_positional_embeddings

class Block(nn.Module):
  def __init__(self, attention_type, embed_dim, mlp_ratio=4, **attn_kwargs):
    super().__init__()
    self.layer_norm1 = nn.LayerNorm(embed_dim)
    self.layer_norm2 = nn.LayerNorm(embed_dim)

    if attention_type == 'MaSSA':
      self.attention = MaSSA(embed_dim, **attn_kwargs)
    else:
      self.attention = MHSA(embed_dim,**attn_kwargs)
    
    self.MLP = MLP(embed_dim, mlp_ratio * embed_dim)

  def forward(self, x):
      h = self.attention(self.layer_norm1(x))
      h = h + x
      c = self.MLP(self.layer_norm2(h))
      c = c + h
      return c
  
class ViT(nn.Module):
  def __init__(self, attention_type,attn_params, embed_dim, nb_blocks, patch_size, nb_classes=10):
    super().__init__()
    self.attention_type = attention_type
    num_patches = (28 // patch_size) ** 2
    self.class_token = nn.Parameter(torch.zeros(1, embed_dim))
    self.pos_embed = nn.Parameter(get_positional_embeddings(num_patches+1, embed_dim))
    self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)

    blocks = []
    for _ in range(nb_blocks):
      blocks.append(
          Block(self.attention_type, embed_dim, **attn_params)
      )
      
    self.blocks = nn.Sequential(*blocks)
    self.norm = nn.LayerNorm(embed_dim)
    self.head = nn.Sequential(
      nn.Linear(embed_dim, embed_dim),
      nn.Tanh(),
      nn.Linear(embed_dim, nb_classes)
    )

  def forward(self, x):
    B, C, H, W = x.shape
    ## image to patches
    x = self.patch_embed(x)
    ## concatenating class token
    cls_token = self.class_token.unsqueeze(0)
    cls_token = cls_token.expand(B, -1, -1)
    x = torch.cat([cls_token,x], dim=1)
    ## adding positional embedding
    x = x + self.pos_embed  # broadcasting sur le batch_size
    ## forward in the transformer
    x = self.blocks(x)
    ## Normalize the output
    x = self.norm(x)
    ## classification output
    output = self.head(x[:,0,:]) # Only CLS
    return output


