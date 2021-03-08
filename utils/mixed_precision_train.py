from __future__ import print_function, division

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

import time
import os
import copy

# from tqdm.auto import tqdm
from tqdm import tqdm

def mixed_precision_train(dataloaders, model, criterion, optimizer, scheduler,  dataset_sizes, device="cpu", num_epochs=1, train=True):
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
                
                with torch.set_grad_enabled(phase=='train'):
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)

                        scaler.update()

                # statistics
                if phase != "train":
                    y_pred = np.append(y_pred, preds.detach().cpu().numpy())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase != 'train':
                epoch_precisions, epoch_recalls, epoch_f1s, _ = precision_recall_fscore_support(y_true, y_pred)
                epoch_precision, epoch_recall, epoch_f1 = np.mean(epoch_precisions), np.mean(epoch_recalls), np.mean(epoch_f1s)
                epoch_bar.write('{} Loss: {:.4f} Acc: {:.4f} Precisions: {:.4f} Recalls: {:.4f} F1: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1))
            else:
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
    if phase == 'train':
        return model
    else: 
        return model, epoch_f1s


def mixed_precision_train_with_tensorboard(name, dataloaders, model, criterion, optimizer, scheduler,  dataset_sizes, device="cpu", num_epochs=1, train=True):
    dataset_sizes = dataset_sizes #{phase: len(dataloader) for phase, dataloader in dataloaders.items()}

    if train:
        phases = ["train", "val"]
    else:
        phases = ["test"]
        num_epochs = 1

    writer = SummaryWriter("runs/" + name)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_model_wts_f1 = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    epoch_bar = tqdm(range(num_epochs), desc="Epoch", position=0)
    scaler = GradScaler()

    batch_cnt = 0
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
                
                with torch.set_grad_enabled(phase=='train'):
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)

                        scaler.update()

                # statistics
                if phase != "train":
                    y_pred = np.append(y_pred, preds.detach().cpu().numpy())

                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                batch_size = len(y_true)

                # writer.add_scalar(phase+"/batch_loss", batch_loss / batch_size, batch_cnt)
                # writer.add_scalar(phase+"/batch_acc", batch_corrects / batch_size, batch_cnt)
                # # writer.add_scalars(phase, {
                # #     "batch_loss": batch_loss / batch_size,
                # #     "batch_acc":  batch_corrects / batch_size,
                # # }, batch_cnt)
                # # writer.close()
                # batch_cnt += 1

                running_loss += batch_loss
                running_corrects += batch_corrects

            if phase == 'train':
                scheduler.step()

            if phase != 'train':
                epoch_precisions, epoch_recalls, epoch_f1s, _ = precision_recall_fscore_support(y_true, y_pred)
                epoch_precision, epoch_recall, epoch_f1 = np.mean(epoch_precisions), np.mean(epoch_recalls), np.mean(epoch_f1s)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            writer.add_scalar(phase+"/epoch_loss", epoch_loss, epoch)
            writer.add_scalar(phase+"/epoch_acc", epoch_acc, epoch)
            if phase != 'train':
                writer.add_scalar(phase+"/epoch_precision", epoch_precision, epoch)
                writer.add_scalar(phase+"/epoch_recall", epoch_recall, epoch)
                writer.add_scalar(phase+"/epoch_f1", epoch_f1, epoch)
                epoch_bar.write('{} Loss: {:.4f} Acc: {:.4f} Precisions: {:.4f} Recalls: {:.4f} F1: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1))
            else:
                epoch_bar.write('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            writer.close()
            # writer.add_scalars(phase, {
            #     "epoch_loss": epoch_loss,
            #     "epoch_acc": epoch_acc,
            #     "epoch_precision": epoch_precision,
            #     "epoch_recall": epoch_recall,
            #     "epoch_f1": epoch_f1,
            # }, epoch)
            # writer.close()
                

            # if phase != "train":
            #     plot_confusion_matrix(y_true, y_pred)
            #     print(classification_report(y_true, y_pred, labels=list(range(y_true.max().astype(np.uint32) + 1))))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts_f1 = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # writer.close()

    torch.save(best_model_wts, "./model/{}_best_val".format(name))
    torch.save(best_model_wts_f1, "./model/{}_best_f1".format(name))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
'''
def train_evaluation(dataloaders, model, criterion, optimizer, scheduler,  dataset_sizes, device="cpu", num_epochs=1):

    for _ in range(num_epochs):
      for inputs, labels in train_loader:
        # move data to proper dtype and device
        inputs = inputs.to(dtype=dtype, device=device)
        labels = labels.to(device=device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
'''