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
from thop import profile
import time


class MHSA(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.W_q = nn.Linear(embed_dim,embed_dim)
        self.W_k = nn.Linear(embed_dim,embed_dim)
        self.W_v = nn.Linear(embed_dim,embed_dim)
        self.projection = nn.Linear(embed_dim,embed_dim)


    @staticmethod
    def attention(q, k, v,scale):
        attention = torch.matmul(q, k.transpose(-2, -1)) / scale
        attention = attention.softmax(dim=-1)
        return torch.matmul(attention, v)


    def forward(self, x):

        B, N, C = x.shape # [Batch_size,Nb_patch + 1 ,Emb_dim]

        # q,k,v [Batch_size,Nb_patch,Emb_dim]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # q,k,v [Batch_size,Nb_patch + 1,Num_heads,Head_dim]

        q = q.reshape(B,N,self.num_heads,self.head_dim)
        k = k.reshape(B,N,self.num_heads,self.head_dim)
        v = v.reshape(B,N,self.num_heads,self.head_dim)

        # q,k,v [Batch_size, Num_heads, Nb_patchs + 1, Head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        h = self.attention(q, k, v, self.scale)  # [B, num_heads, N, head_dim]
        h = h.permute(0, 2, 1, 3).contiguous()  # [B, N, num_heads, head_dim]
        concat_h = h.reshape(B, N, C)  # [B, N, embed_dim] 

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
        
        return self.projection(out), context_vectors


def time_model(model, x, n_warmup=10, n_runs=50):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        torch.cuda.synchronize()
        
        t0 = time.time()
        for _ in range(n_runs):
            _ = model(x)
        torch.cuda.synchronize()

    return (time.time() - t0) / n_runs


if __name__ == "__main__":
    embed_dim = 128
    num_heads = 4
    k = 4
    gamma = 0.5
    batch_size = 16
    N = [128, 256, 512, 1024, 2048]

    for nb_tokens in N:
        print(f"\n Number of tokens: {nb_tokens}")
        x = torch.randn(batch_size, nb_tokens, embed_dim).cuda()

        mhsa = MHSA(embed_dim, num_heads).cuda()
        massa = MaSSA(embed_dim, k, gamma).cuda()

        mhsa_time = time_model(mhsa, x)
        massa_time = time_model(massa, x)

        print(f"MHSA  Time: {mhsa_time*1000:.3f} ms")
        print(f"MaSSA Time: {massa_time*1000:.3f} ms")
        print(f"Speedup: {mhsa_time / massa_time:.2f}x")