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
                transforms.Resize((image_size,image_size)),
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
    
    train_dataset.set_transform =   transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
    ])
    

    return train_dataset, val_dataset, test_dataset
