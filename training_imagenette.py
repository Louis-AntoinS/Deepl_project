import torch.cuda as cuda
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.utils as vutils
import torchvision
from src.models.match_vit import MatchVit
from src.models.mobilevit import MobileViT
from src.models.vit import ViT
from thop import profile,clever_format
from src.utils import train_test

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.7, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.Imagenette('./data', split ='train', download=True, transform=train_transform)
test_dataset = torchvision.datasets.Imagenette('./data', split ='val', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("classes : ",len(train_dataset.classes))
print("train : ", len(train_dataset))
print("test : ", len(test_dataset)) 


lr = 1e-4   
EPOCHS = 20
embed_dim = 192
patch_size = 16
nb_blocks = 3
img_size = 256
img_chan = 3

exemple_input = torch.randn(1, 3, 256, 256).to(device)

#print("------ Training MatchVIT ------")

model_MatchVIT = MatchVit(num_classes=10, alpha=1.5).to(device)
model_Macs, model_params = profile(model_MatchVIT, inputs=(exemple_input,))
model_Macs_readable, model_params_readable = clever_format([model_Macs, model_params], "%.3f")
print(f"Formatted MatchVIT MACs: {model_Macs_readable}, Formatted Parameters: {model_params_readable}")
opt_MatchVIT = torch.optim.Adam(model_MatchVIT.parameters(), lr=1e-4, weight_decay=1e-2)
train_test(device,train_loader,test_loader,model_MatchVIT,opt_MatchVIT,EPOCHS)
del model_MatchVIT
cuda.empty_cache()


#print("------ Training MobileViT ------")
model_ViT = MobileViT((256, 256), dims = [144, 192, 240], channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640], num_classes=10).to(device)
opt_ViT = torch.optim.Adam(model_ViT.parameters(), lr=lr, weight_decay=1e-2)
train_test(device,train_loader,test_loader,model_ViT,opt_ViT,EPOCHS)

model_Macs, model_params = profile(model_ViT, inputs=(exemple_input,))
model_Macs_readable, model_params_readable = clever_format([model_Macs, model_params], "%.3f")
print(f"Formatted ViT MACs: {model_Macs_readable}, Formatted Parameters: {model_params_readable}")