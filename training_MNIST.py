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
from src.modules.layers import MLP, PatchEmbed, get_positional_embeddings
from src.models.vit import ViT
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def count_parameters(model):
    """
    Count trainable vs total parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def eval_model(model, loader, device): 
    model.eval()
    acc = 0.
    c = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yhat = model(x)

        acc += torch.sum(yhat.argmax(dim=1) == y).item()
        c += len(x)

    model.train()
    return round(100 * acc / c, 2)

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

model = ViT('MHSA', {'num_heads': 4}, embed_dim, nb_blocks = nb_blocks, patch_size = patch_size).to(device)
"""model = ViT('MaSSA', {'k':4, 'gamma':0.5}, embed_dim, nb_blocks = nb_blocks, patch_size = patch_size).to(device)"""
opt = torch.optim.Adam(model.parameters(), lr=lr)

print(f"Model has {count_parameters(model)} trainable parameters.")

for epoch in range(EPOCHS):
  train_loss = 0.
  for x, y in train_loader:
    x, y = x.to(device), y.to(device)

    opt.zero_grad()
    yhat = model(x)
    loss = F.cross_entropy(yhat, y)
    loss.backward()

    opt.step()

    train_loss += loss.item()

  print(f"--- Epoch {epoch} ---")
  print(f"Train loss: {train_loss / len(train_loader)}")

acc = eval_model(model, test_loader, device)
print(f"Test accuracy: {acc}")




