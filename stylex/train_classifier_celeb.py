import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
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
img_size = 128# 3 #64
celeb_train, celeb_val, celeb_test = get_train_valid_test_dataset("../data/CelebA/celeba-dataset/", "../data/CelebA/celeba-dataset/list_attr_celeba.csv", "Male", image_size=img_size)            


batch_size = 32
cel_train_loader = DataLoader(celeb_train, batch_size=batch_size, pin_memory=True)
cel_val_loader = DataLoader(celeb_val, batch_size=batch_size)
cel_test_loader = DataLoader(celeb_test, batch_size=batch_size)
print("Dataset preparado")
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

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
            loss += criterion(output, target).item() * len(target)/batch_size

            # Get the predictions.
            preds = torch.argmax(output, 1)

            # Calculate the accuracy.
            accuracy += torch.sum(preds == target).item() * len(target)/batch_size

    # Calculate the average loss and accuracy.
    loss /= len(loader)
    accuracy /= len(loader) * batch_size

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
train_model(model, cel_train_loader, cel_val_loader, optimizer, criterion, num_epochs=10)

# Test the model.
test_loss, test_acc = validate_model(model, cel_test_loader, criterion)

# Print the test loss.
print('Test Loss: {:.4f}'.format(test_loss))

# Print the test accuracy.
print('Test Accuracy: {:.4f}'.format(test_acc))    
torch.save(model.state_dict(), './resnet-18-'+str(img_size)+ 'px-age-classifier_full_10epochs_acc_'+str(test_acc)+'.pt')