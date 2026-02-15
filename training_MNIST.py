from IPython import embed
import torch
from torch import mode, nn
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
from src.models.vit import ViT, Block, MaSSA_Block, PatchEmbed, get_positional_embeddings
from src.utils import train_test, eval_model, count_parameters
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from thop import profile

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,)) # Normalisation MNIST
])


train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.001
EPOCHS = 10
embed_dim = 128
patch_size = 7
nb_blocks = 2
img_size = 28
img_chan = 1


print("------ Training ViT ------")
model_ViT = ViT({'num_heads': 4}, embed_dim, nb_blocks = nb_blocks, patch_size = patch_size,img_size=img_size,img_chan=img_chan, use_matchvit_components=False).to(device)
opt_ViT = torch.optim.Adam(model_ViT.parameters(), lr=lr)
"""train_test(device,train_loader,test_loader,model_ViT,opt_ViT,EPOCHS)"""

print("------ Training MatchVIT ------")
model_MatchVIT = ViT({'k':4, 'gamma':0.5}, embed_dim, nb_blocks = nb_blocks, patch_size = patch_size,img_size=img_size,img_chan=img_chan,use_matchvit_components=True).to(device)
opt_MatchVIT = torch.optim.Adam(model_MatchVIT.parameters(), lr=lr)
"""train_test(device,train_loader,test_loader,model_MatchVIT,opt_MatchVIT,EPOCHS)"""



