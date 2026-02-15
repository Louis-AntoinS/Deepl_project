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


class MLP(nn.Module):
  def __init__(self, embed_dim, expansion_factor = 4):
    super().__init__()

    self.hidden_dim = expansion_factor * embed_dim
    
    self.layer = nn.Sequential(
        nn.Linear(embed_dim, self.hidden_dim),
        nn.GELU(),
        nn.Linear(self.hidden_dim, embed_dim)
    )

  def forward(self, x):
    return self.layer(x)
  

class ContextGatedFFN(nn.Module):
    def __init__(self, embed_dim, k, gamma, expansion_factor = 4):
        super().__init__()
        
        self.k = k 
        self.reduced_dim = int(embed_dim * gamma) # γC (dimension réduite de MaSSA)
        self.hidden_dim = expansion_factor * embed_dim # λC

        # Branche du bas
        self.W_1 = nn.Linear(embed_dim, self.hidden_dim)   
        self.W_2 = nn.Linear(self.hidden_dim, embed_dim)   

        # Branche du haut 
        self.W_k = nn.Linear(embed_dim, self.k)            
        self.W_cv = nn.Linear(self.reduced_dim, self.hidden_dim)  
        self.silu = nn.SiLU()

    def forward(self, x, context_vectors):
        
        cv = context_vectors.detach() 
        x_k = self.W_k(x)                   # (B, N, k) 
        cv_proj = self.W_cv(cv)             # (B, k, λC) 
        gate = torch.matmul(x_k, cv_proj) 
        gate = self.silu(gate)              # σ(xWk cv Wcv)
        out = self.silu(self.W_1(x))        # σ(xW1)
        out = out * gate  
        
        return self.W_2(out)           
                   
                