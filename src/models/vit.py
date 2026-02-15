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
from src.modules.ffn import MLP, ContextGatedFFN


class PatchEmbed(nn.Module):

  def __init__(self, in_chan=1, patch_size=7, embed_dim=128):
    super().__init__()

    self.projection = nn.Conv2d(
              in_channels=in_chan,
              out_channels=embed_dim,
              kernel_size=patch_size,
              stride=patch_size
    )

  def forward(self, x):
      # x (batch,nb_patch)
      x = self.projection(x)
      B, C, H, W = x.shape       # (Batch_size,embed_dim,h,w)
      x = x.reshape(B,C,-1)      # (Batch_size,embed_dim,Nb_patchs)
      x = x.permute(0,2,1)       # (Batch_size,Nb_patchs,embed_dim)
      return x  # B, N, C
  
  
def get_positional_embeddings(sequence_length, d):
    result = torch.ones(1,sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result



class Block(nn.Module):
  def __init__(self, embed_dim, expansion_factor=4, **attn_kwargs):
    super().__init__()
    self.layer_norm1 = nn.LayerNorm(embed_dim)
    self.layer_norm2 = nn.LayerNorm(embed_dim)
    num_head = attn_kwargs['num_heads']
    self.attention = MHSA(embed_dim, num_head)
    self.MLP = MLP(embed_dim, expansion_factor)

  def forward(self, x):
      h = self.attention(self.layer_norm1(x))
      h = h + x
      c = self.MLP(self.layer_norm2(h))
      c = c + h
      return c
  
  
class MaSSA_Block(nn.Module):
  def __init__(self, embed_dim, expansion_factor=4, **attn_kwargs):
    super().__init__()
    self.layer_norm1 = nn.LayerNorm(embed_dim)
    self.layer_norm2 = nn.LayerNorm(embed_dim)
    gamma = attn_kwargs['gamma']
    k = attn_kwargs['k']
    self.attention = MaSSA(embed_dim,k, gamma)
    self.context_gated_ffn = ContextGatedFFN(embed_dim, k, gamma, expansion_factor)

  def forward(self, x):
      h, context_vectors = self.attention(self.layer_norm1(x))
      h = h + x
      c = self.context_gated_ffn(self.layer_norm2(h), context_vectors)
      c = c + h
      return c


class ViT(nn.Module):
  def __init__(self,attn_params, embed_dim, nb_blocks, patch_size, nb_classes=10, img_size=28,img_chan=1, use_matchvit_components=True):
    super().__init__()

    self.embed_dim = embed_dim
    num_patches = (img_size // patch_size) ** 2
    self.class_token = nn.Parameter(torch.zeros(1, embed_dim))
    self.pos_embed = nn.Parameter(get_positional_embeddings(num_patches+1, embed_dim))
    self.patch_embed = PatchEmbed(in_chan= img_chan, patch_size=patch_size, embed_dim=embed_dim)

    if use_matchvit_components:
       block_class = MaSSA_Block
    else:
        block_class = Block
    
    blocks = []
    for _ in range(nb_blocks):
      blocks.append(
          block_class(embed_dim, **attn_params)
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
    x = self.patch_embed(x)
    cls_token = self.class_token.unsqueeze(0)
    cls_token = cls_token.expand(B, -1, -1)
    x = torch.cat([cls_token,x], dim=1)
    x = x + self.pos_embed 
    x = self.blocks(x)
    x = self.norm(x)
    output = self.head(x[:,0,:])
    return output



