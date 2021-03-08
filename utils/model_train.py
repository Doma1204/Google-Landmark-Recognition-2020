from __future__ import print_function, division

import torch
from torch.cuda.amp import GradScaler
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix

import time
import os
import copy

from tqdm.auto import tqdm

def train_model(dataloaders, model, criterion, optimizer, scheduler,  dataset_sizes, device="cpu", num_epochs=1, train=True):
    dataset_sizes = dataset_sizes #{phase: len(dataloader) for phase, dataloader in dataloaders.items()}

    if train:
        phases = ["train", "val"]
    else:
        phases = ["test"]
        num_epochs = 1

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_bar = tqdm(range(num_epochs), desc="Epoch", position=0)
    scaler = GradScaler()

    for epoch in epoch_bar:
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        epoch_bar.write('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                y_pred, y_true = np.array([]), np.array([])

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.

            for inputs, labels in tqdm(dataloaders[phase], desc=phase, position=1, leave=False):

                if phase != "train":
                    y_true = np.append(y_true, np.array(labels))
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if torch.cuda.is_available():
                    model.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                if phase != "train":
                    y_pred = np.append(y_pred, preds.detach().cpu().numpy())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            epoch_bar.write('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # if phase != "train":
            #     plot_confusion_matrix(y_true, y_pred)
            #     print(classification_report(y_true, y_pred, labels=list(range(y_true.max().astype(np.uint32) + 1))))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model