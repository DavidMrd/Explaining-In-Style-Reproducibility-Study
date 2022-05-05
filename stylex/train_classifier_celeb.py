import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

from torch.utils.data import Dataset, DataLoader
from resnet_classifier import load_resnet_classifier

device = "cuda:0"
model_name = "resnet-18-64px-gender.pt"
print("Cargando model"+str(model_name)+" ...")
cuda_device = 0
n_outputs = 2 
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
model.fc = nn.Linear(512, n_outputs).to(device)
print(model)
print("Cargado")

import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
from torchvision import transforms
from CelebA_utils import get_train_valid_test_dataset
img_size = 3 #64
celeb_train, celeb_val, celeb_test = get_train_valid_test_dataset("../data/CelebA/celeba-dataset/", "../data/CelebA/celeba-dataset/list_attr_celeba.csv", "Male", image_size=img_size)            

torch.cuda.empty_cache()
batch_size = 128
cel_train_loader = DataLoader(celeb_train, batch_size=batch_size, pin_memory=True)
cel_val_loader = DataLoader(celeb_val, batch_size=batch_size)
cel_test_loader = DataLoader(celeb_test, batch_size=batch_size)
print("Dataset preparado")
optimizer = optim.Adam(model.parameters(), lr=0.0001)
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
            loss += criterion(output, target).item() * len(target)/128

            # Get the predictions.
            preds = torch.argmax(output, 1)

            # Calculate the accuracy.
            accuracy += torch.sum(preds == target).item() * len(target)/128

    # Calculate the average loss and accuracy.
    loss /= len(loader)
    accuracy /= len(loader) * batch_size

    return loss, accuracy

def train_model(model, train_loader, val_loader, optimizer, criterion, test_model=False, epochs=10):
    """Trains model"""

    # Put the model in training mode.
    model.train()

    # For each epoch...
    for epoch in range(epochs):
        # For each batch in the training set...
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            # Send the data and labels to the device.
            data, target = data.to(device), target.to(device)

            # Zero out the gradients.
            optimizer.zero_grad()
            print(data.shape)
            # Forward pass.
            output = model(data)

            # Calculate the loss.
            loss = criterion(output, target)

            # Backward pass.
            loss.backward()

            # Update the weights.
            optimizer.step()

            # Print the loss.
            if batch_idx % 100 == 0:
                print('Epoch: {}/{}'.format(epoch + 1, epochs),
                      'Loss: {:.4f}'.format(loss.item()))
        if((epoch%3)==0):
            # Validate the model.
            val_loss, val_acc = validate_model(model, val_loader, criterion)

            # Print the validation loss.
            print('Validation Loss: {:.4f}'.format(val_loss))

            # Print the validation accuracy.
            print('Validation Accuracy: {:.4f}'.format(val_acc))

    if test_model:
        # Test the model.
        test_loss, test_acc = validate_model(model, test_loader, criterion)

        # Print the test loss.
        print('Test Loss: {:.4f}'.format(test_loss))

        # Print the test accuracy.
        print('Test Accuracy: {:.4f}'.format(test_acc))    

model.requires_grad_(True)
train_model(model, cel_train_loader, cel_val_loader, optimizer, criterion, epochs=10)

# Test the model.
test_loss, test_acc = validate_model(model, cel_test_loader, criterion)

# Print the test loss.
print('Test Loss: {:.4f}'.format(test_loss))

# Print the test accuracy.
print('Test Accuracy: {:.4f}'.format(test_acc))    
torch.save(model.state_dict(), './resnet-18-'+str(img_size)+ 'px-age-classifier_full_10epochs_acc_'+str(test_acc)+'.pt')