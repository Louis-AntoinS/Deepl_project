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
import time
from tqdm import tqdm
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

    for x, y in tqdm(loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        yhat = model(x)

        acc += torch.sum(yhat.argmax(dim=1) == y).item()
        c += len(x)

    model.train()
    return round(100 * acc / c, 2)


def train_test(device,train_loader,test_loader,model,optimizer,epochs):
    
    print(f"Model has {count_parameters(model)} trainable parameters.")
    
    for epoch in range(epochs):
        train_loss = 0.
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            yhat = model(x)
            loss = F.cross_entropy(yhat, y)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            del x, y, yhat, loss
            torch.cuda.empty_cache()
        print(f"--- Epoch {epoch} ---")
        print(f"Train loss: {train_loss / len(train_loader)}")
    acc = eval_model(model, test_loader, device)
    print(f"Test accuracy: {acc}")
    
            