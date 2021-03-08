from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset
from efficientnet_pytorch import EfficientNet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import time
import os
import copy

from PIL import Image
from tqdm import tqdm

from utils.data_utils import get_dataloader, get_dataloader_2
from utils.loss import LDAMLoss
from utils.model_train import train_model
from utils.mixed_precision_train import mixed_precision_train_with_tensorboard

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_preprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform = {
    "train": train_preprocess,
    "val": preprocess,
    "test": preprocess,
}

dataloaders, dataset_size, num_classes, distributions = get_dataloader_2("./data/train_discard_6.csv", "./data/train_224/", transform=transform, batch_size=64, shuffle=True, num_workers=32)

model_name = "EfficientNet-B0_discard_6_batch64_epoch5_LDAM_sgd_aug_tensorboard"

model = EfficientNet.from_pretrained("efficientnet-b0")
#model = models.resnet50(pretrained=True)
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, num_classes["train"])

# criterion = nn.CrossEntropyLoss()
criterion = LDAMLoss(distributions["train"], max_m=0.5, s=30)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model = model.to(device)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = mixed_precision_train_with_tensorboard(model_name, dataloaders, model, criterion, optimizer, exp_lr_scheduler,dataset_size, num_epochs=5, device=device)
model = mixed_precision_train_with_tensorboard(model_name, dataloaders, model, criterion, optimizer, exp_lr_scheduler,dataset_size, device=device, train=False)


