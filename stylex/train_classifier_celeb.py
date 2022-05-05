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
model = load_resnet_classifier(model_name, cuda_device,n_outputs)
print(model)
print("Cargado")

import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
from torchvision import transforms

class CelebA(data.Dataset):
    def __init__(self, celeb_dir, csv_path, image_size=32, transform=None, label="male"):
        """
        PyTorch DataSet for the FFHQ-Age dataset.
        :param root: Root folder that contains a directory for the dataset and the csv with labels in the root directory.
        :param label: Label we want to train on, chosen from the csv labels list.
        """
        self.target_class = label

        # Store image paths
        image_path = os.path.join(celeb_dir, "img_align_celeba", "img_align_celeba")
        self.images = [os.path.join(image_path, file)
                       for file in os.listdir(image_path) if file.endswith('.jpg')]

        # Import labels from a CSV file
        self.labels = pd.read_csv(csv_path)

        # Image transformation
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Make a lookup dictionary for the labels
        # Get column names of dataframe
        cols = self.labels.columns.values
        label_ids = {col_name: i for i, col_name in enumerate(cols)}
        self.class_id = label_ids[self.target_class]

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        _img = self.transform(Image.open(self.images[index]))
        _label = 0 if self.labels.iloc[index, self.class_id] == 1 else 1  # Male will be the first number as with FFHQ upstairs
        return _img, _label

    def __len__(self):
        return len(self.images)



def get_train_valid_test_dataset(celeba_dir, csv_path, label, image_size=32, valid_ratio=0.15, test_ratio=0.15):
    # TODO: Specify different training routines here per class (such as random crop, random horizontal flip, etc.)

    dataset = CelebA(celeba_dir, csv_path, image_size=image_size, label=label)
    train_length, valid_length, test_length = int(len(dataset) * (1 - valid_ratio - test_ratio)), \
                                              int(len(dataset) * valid_ratio), int(len(dataset) * test_ratio)
    # Make sure that the lengths sum to the total length of the dataset
    remainder = len(dataset) - train_length - valid_length - test_length
    train_length += remainder
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                             [train_length, valid_length, test_length],
                                                                             generator=torch.Generator().manual_seed(42)
                                                                             )
    """
    train_dataset.set_transform = A.Compose(
        [
            transforms.Resize(image_size),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.HorizontalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.25, contrast_limit=0.5),
            A.MotionBlur(p=.2),
            A.GaussNoise(p=.2),
            A.ImageCompression(p=.2, quality_lower=50),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            transforms.Resize(224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    """

    return train_dataset, val_dataset, test_dataset

img_size = 128 #64
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
torch.save(model.state_dict(), './resnet-18-'+img_size+ 'px-age-classifier_full_10epochs_acc_'+str(test_acc)+'.pt')