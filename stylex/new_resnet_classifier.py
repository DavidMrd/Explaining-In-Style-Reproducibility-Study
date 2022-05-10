# Imports
# Source https://www.kaggle.com/code/scratchpad/notebookc5e5d9859d/edit
import os
import sys
import time
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import Namespace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt

# Paths
celeba_dir = "../data/CelebA/celeba-dataset/"
attr_path = f"{celeba_dir}list_attr_celeba.csv"
img_dir = f"{celeba_dir}img_align_celeba/"

# Hyperparameters (and other variables)
hp = Namespace(
    image_size = 64,  # ResNet needs size multiple of 32
    dropout = 0,
    optimizer = "Adam",
    lr = 1e-3,
    momentum = .5,
    weight_decay = 0,
    num_epochs = 50,
    batch_size = 32,
    perc_tr = .7,
    resnet_version = 18,
    device = "cuda" if torch.cuda.is_available() else "cpu"
)

# Downloads and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((hp.image_size, hp.image_size)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

img_dataset = ImageFolder(img_dir, transform)

subset_idx = int(len(img_dataset)*0.5)
img_subset = torch.utils.data.Subset(img_dataset, range(subset_idx))

labels = pd.read_csv(attr_path)
labels = torch.FloatTensor(np.array([l==1 for l in labels["Male"]]))

images_subset = torch.stack([x for x,_ in tqdm(img_subset)])
labels_subset = labels[:subset_idx]

data_subset = data.TensorDataset(images_subset, labels_subset)

n = len(data_subset)
tr_dsubset, va_dsubset = data.random_split(data_subset, [int(n*hp.perc_tr),n-int(n*hp.perc_tr)],
                                           torch.Generator().manual_seed(1234))

tr_dloader = data.DataLoader(tr_dsubset, batch_size=hp.batch_size, shuffle=True,num_workers=2, drop_last=True)
va_dloader = data.DataLoader(va_dsubset, batch_size=hp.batch_size, shuffle=True,num_workers=2, drop_last=True)

dloader_dict = {'train': tr_dloader, 'valid': va_dloader}

class feature_extractor(nn.Module):
    def __init__(self, resnet_version, coupling_size):
        super(feature_extractor,self).__init__()
        self.__load_resnet__(resnet_version)
        self.resnet.fc = nn.Linear(512, coupling_size)

    def forward(self, x):
        return self.resnet(x)
    
    def __load_resnet__(self, version):
        if type(version) is str:
            version = int(version)
        elif type(version) not in [int, str]:
            sys.exit("ResNet version argument must be an integer or string in [18, 34, 50, 101, 152].")
        if version == 18:
            self.resnet = models.resnet18()
        elif version == 34:
            self.resnet = models.resnet34()
        elif version == 50:
            self.resnet = models.resnet54()
        elif version == 101:
            self.resnet = models.resnet101()
        elif version == 152:
            self.resnet = models.resnet152()
        else:
            sys.exit(f"""ResNet{version} is not available.\n
                         Choose one of [18, 34, 50, 101, 152].""")

class classifier(nn.Module):
    def __init__(self, coupling_size, dropout):
        super(classifier,self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(coupling_size, 128),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.MLP(x)
        return self.sigmoid(x).squeeze(1)

class GenRec(nn.Module):
    def __init__(self, resnet_version=18, coupling_size=512, dropout=0):
        super(GenRec,self).__init__()
        torch.manual_seed(1234)
        self.best_acc = 0
        self.FE = feature_extractor(resnet_version, coupling_size)
        self.MLP = classifier(coupling_size, dropout)

    def forward(self, x, state=None):
        x = self.FE(x)
        x = self.MLP(x)
        return x

    def num_params(self, trainable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable)
    
    def train_model(self, dloaders, optimizer, hparams, wb_run=None, prints=False):
        loss_fn = nn.BCELoss()
        wandb.watch(self, loss_fn, log="all", log_freq=50) if wb_run else None
        va_losses, va_accs, tr_losses, tr_accs = [], [], [], []
        best_model = copy.deepcopy(self.state_dict())
        for epoch in range(hparams.num_epochs):
            print(f"Epoch {epoch+1}/{hparams.num_epochs}") if prints else None
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()
                accum_loss, accum_acc = 0, 0
                for x, y in dloaders[phase]:
                    x, y = x.to(hparams.device), y.to(hparams.device)
                    optimizer.zero_grad()
                    out = self.forward(x)
                    loss = loss_fn(out, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    # Training stats
                    accum_loss += loss.item() * x.size(0)
                    preds = torch.round(out).detach()
                    accum_acc += torch.sum(preds == y).item()
                n = len(dloaders[phase].dataset)
                epoch_loss, epoch_acc = accum_loss/n, accum_acc/n
                if prints:
                    print('[{}] Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                if wb_run:
                    wandb.log({f"{phase}_loss":epoch_loss, f"{phase}_acc":epoch_acc}, step=epoch)
                if phase == "valid":
                    va_losses.append(epoch_loss); va_accs.append(epoch_acc)
                    if self.best_acc < epoch_acc:
                        self.best_acc = epoch_acc
                        best_model = copy.deepcopy(self.state_dict())
                else:
                    tr_losses.append(epoch_loss); tr_accs.append(epoch_acc)
            self.load_state_dict(best_model)
        return {"loss":va_losses, "acc":va_accs}, {"loss":tr_losses, "acc":tr_accs}

# Training setup
gr_mod = GenRec(dropout=hp.dropout).to(hp.device)

if hp.optimizer == "Adam":
    optimizer = optim.Adam(gr_mod.parameters(), lr=hp.lr, amsgrad=True)
elif hp.optimizer == "SGD":
    optimizer = torch.optim.SGD(gr_mod.parameters(), lr=hp.lr,
                                momentum=hp.momentum, weight_decay=hp.weight_decay)

config = dict(
    optimizer = "Adam",
    lr = 1e-3,
    momentum = .5,
    weight_decay = 0,
    resnet_version = 18,
    dropout = 0
)         

# Run this if you are not using W&B
va_info, tr_info = gr_mod.train_model(dloader_dict, optimizer, hp, prints=True)

# Save model's parameters to a file
torch.save(gr_mod.state_dict(), "/kaggle/working/model_checkpoint.pth")


# Showcase performace
import torchvision.utils as utils
gr_mod.to(hp.device)
for x,y in va_dloader:
    y = y.detach().numpy()
    preds = torch.round(gr_mod.forward(x))
    print(f"acc = {sum([preds[i]==y[i] for i in range(len(preds))])/len(preds)}")
    str_pred = ["man" if r else "woman" for r in preds]
    grid = utils.make_grid(x, nrow=8)
    plt.imshow(grid.permute(1, 2, 0))
    print(str_pred)
    break