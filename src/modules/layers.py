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
      B, C, H, W = x.shape # (Batch_size,embed_dim,h,w)
      x = x.reshape(B,C,-1) # (Batch_size,embed_dim,Nb_patchs)
      x = x.permute(0,2,1) # (Batch_size,Nb_patchs,embed_dim)
      # x.shape() -> B, N, C
      return x

class MLP(nn.Module):
  def __init__(self, in_features, hid_features):
    super().__init__()

    self.layer = nn.Sequential(
        nn.Linear(in_features,hid_features),
        nn.GELU(),
        nn.Linear(hid_features,in_features)
    )

  def forward(self, x):
    return self.layer(x)
  
def get_positional_embeddings(sequence_length, d):
    result = torch.ones(1,sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result