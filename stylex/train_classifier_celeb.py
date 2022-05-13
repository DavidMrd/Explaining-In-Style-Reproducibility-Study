import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import numpy as np

from torch.utils.data import Dataset, DataLoader
from resnet_classifier import load_resnet_classifier
#torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
if(device == "cpu"):
    print("ERROR, not GPU available")

model_name = "resnet-50-64px-gender.pt"
print("Cargando model"+str(model_name)+" ...")
cuda_device = 0
n_outputs = 2 
model = models.resnet50(pretrained=True).to(device)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, n_outputs).to(device)
print(model)
print("Cargado")

import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
from torchvision import transforms
from CelebA_utils import get_train_valid_test_dataset
import time
import copy
#celeb_train, celeb_val, celeb_test = get_train_valid_test_dataset("../data/CelebA/celeba-dataset/", "../data/CelebA/celeba-dataset/list_attr_celeba.csv", "Male", image_size=hp.img_size)            
# Paths
celeba_dir = "../data/CelebA/celeba-dataset/"
attr_path = f"{celeba_dir}list_attr_celeba.csv"
img_dir = f"{celeba_dir}img_align_celeba/"

optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
from torchvision.datasets import ImageFolder
from argparse import Namespace
hp = Namespace(
    image_size = 64,  # ResNet needs size multiple of 32
    dropout = 0,
    optimizer = "Adam",
    lr = 1e-3,
    momentum = .5,
    weight_decay = 0,
    num_epochs = 5,
    batch_size = 32,
    perc_tr = .8,
    resnet_version = 18,
    device = "cuda" if torch.cuda.is_available() else "cpu"
)
# Downloads and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((hp.image_size, hp.image_size)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_dataset = ImageFolder(img_dir, transform)
#images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
max_idx_train = 162770
max_idx_trainval = 182637

trainval_subset_idx = max_idx_trainval#int(len(img_dataset)*0.8)
img_subset = torch.utils.data.Subset(img_dataset, range(trainval_subset_idx))
test_img_subset = torch.utils.data.Subset(img_dataset, range(trainval_subset_idx,len(img_dataset)))
labels = pd.read_csv(attr_path)
labels = torch.LongTensor(np.array([l==1 for l in labels["Male"]]))

images_subset = torch.stack([x for x,_ in tqdm(img_subset)])
test_images_subset = torch.stack([x for x,_ in tqdm(test_img_subset)])

labels_subset = labels[:trainval_subset_idx]
test_labels_subset = labels[trainval_subset_idx:len(img_dataset)]

trainval_data_subset = data.TensorDataset(images_subset, labels_subset)
test_dsubset = data.TensorDataset(test_images_subset, test_labels_subset)
n = len(trainval_data_subset)
tr_dsubset, va_dsubset = data.random_split(trainval_data_subset, [int(n*hp.perc_tr)+1,int(n*(1-hp.perc_tr))],
                                           torch.Generator().manual_seed(1234))


tr_dloader = data.DataLoader(tr_dsubset, batch_size=hp.batch_size, shuffle=True,num_workers=2, drop_last=True)
va_dloader = data.DataLoader(va_dsubset, batch_size=hp.batch_size, shuffle=True,num_workers=2, drop_last=True)
test_dloader = data.DataLoader(test_dsubset, batch_size=hp.batch_size, shuffle=True,num_workers=2, drop_last=True)
print("El numero de imagenes usadas para train es "+str(len(tr_dloader.dataset)))
print("El numero de imagenes usadas para val es "+str(len(va_dloader.dataset)))
print("El numero de imagenes usadas para test es "+str(len(test_dloader.dataset)))

from tqdm import tqdm
def validate_model(model, loader, criterion):
    """Validate the model"""

    # Set the model to evaluation mode.
    model.eval()

    # Initialize the loss and accuracy.
    loss = 0
    accuracy = 0

    # For each batch in the validation set...
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            # Send the batch to the device.

            data, target = data.to(device), target.to(device)

            # Forward pass.
            output = model(data)

            # Calculate the loss.
            loss += criterion(output, target).item() * len(target)/hp.batch_size

            # Get the predictions.
            preds = torch.argmax(output, 1)

            # Calculate the accuracy.
            accuracy += torch.sum(preds == target).item() * len(target)/hp.batch_size

    # Calculate the average loss and accuracy.
    loss /= len(loader)
    accuracy /= len(loader) * hp.batch_size

    return loss, accuracy

def train_model(model, train_loader, val_loader, optimizer, criterion, test_model=False, num_epochs=10):    
    """Trains model"""
    dataloaders = {"train":train_loader,"val":val_loader}
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #for inputs, labels in dataloaders[phase]:
            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):    
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

model.requires_grad_(True)
train_model(model, tr_dloader, va_dloader, optimizer, criterion, num_epochs=hp.num_epochs)

# Test the model.
test_loss, test_acc = validate_model(model, test_dloader, criterion)

# Print the test loss.
print('Test Loss: {:.4f}'.format(test_loss))

# Print the test accuracy.
print('Test Accuracy: {:.4f}'.format(test_acc))    
torch.save(model.state_dict(), './resnet-18-'+str(hp.image_size)+ 'px-gender-classifier_full_10epochs_acc_'+str(test_acc)+'.pt')