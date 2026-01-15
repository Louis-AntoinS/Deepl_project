from re import M
from IPython import embed
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


class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_heads = num_heads

        self.W_q = nn.Linear(embed_dim,embed_dim)
        self.W_k = nn.Linear(embed_dim,embed_dim)
        self.W_v = nn.Linear(embed_dim,embed_dim)
        self.projection = nn.Linear(embed_dim,embed_dim)

    def forward(self, x):
        
        # You need to reshape and permute dimension in a certain manner
        # so that each head (C // num_heads) interact
        # only with its dimensions and not other heads.

        # Try to write at each operation the shape of the tensor if you
        # feel confused.

        B, N, C = x.shape # [Batch_size,Nb_patch + 1 ,Emb_dim]

        # q,k,v [Batch_size,Nb_patch,Emb_dim]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        #q,k,v [Batch_size,Nb_patch + 1,Num_heads,Head_dim]

        q = q.reshape(B,N,self.num_heads,self.head_dim)
        k = k.reshape(B,N,self.num_heads,self.head_dim)
        v = v.reshape(B,N,self.num_heads,self.head_dim)

        #q,k,v [Batch_size, Num_heads, Nb_patchs + 1, Head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        h = []

        for i in range(self.num_heads) :
            attention = torch.matmul(q[:,i,:,:], k[:,i,:,:].transpose(-1, -2)) / self.scale
            attention = F.softmax(attention,dim=-1)
            hi = torch.matmul(attention,v[:,i,:,:])
            h.append(hi)

        # each hi is dim [B, N, head_dim]
        concat_h = torch.cat(h,dim =-1) # [B, N, embed_dim]
        x = self.projection(concat_h)

        return x
    



class MaSSA(nn.Module):
    def __init__(self, embed_dim, k=4, gamma=0.5):
        super().__init__()
        self.k = k
        self.reduced_dim = int(embed_dim * gamma)

        self.W_I = nn.Linear(embed_dim,k)
        self.W_K = nn.Linear(embed_dim,self.reduced_dim)
        self.W_R = nn.Linear(embed_dim,self.reduced_dim)
        self.W_V = nn.Linear(embed_dim,embed_dim)
        self.W_MCV = nn.Linear(self.reduced_dim,embed_dim)
        self.projection = nn.Linear(embed_dim,embed_dim)


    def forward(self, x):
        B, N, C = x.shape # [Batch_size,Nb_patch + 1 ,Emb_dim]

        I = self.W_I(x) # [B, N, k]
        K = self.W_K(x) # [B, N, γC]
        R = self.W_R(x) # [B, N, γC]
        V = self.W_V(x) # [B, N, C]

        # Compute context score
        context_scores = F.softmax(I, dim=1)

        # Compute  k context vectors
        context_vectors = torch.matmul(context_scores.transpose(-1,-2), K)  # [B, k, γC]

        # Compute Match Score 

        match_scores = torch.matmul(R, context_vectors.transpose(-1,-2))  # [B, N, k]
        match_scores = F.softmax(match_scores, dim=-1)

        # Compute MCV
        mcv = torch.matmul(match_scores, context_vectors)  # [B, N, γC]
        mcv = self.W_MCV(mcv)  # [B, N, C]
        out = mcv * V  # [B, N, C]

        return self.projection(out)
    

    