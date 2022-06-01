import os
import sys
import time
from datetime import datetime
import scipy.io as io
from augmentation_tCNN import *
from dataloader import *
from confusion import *
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import scipy
import pandas as pd
import gc
import numpy as np
from DiceNet import * 
from utils import *
import argparse


# control randomness
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()


# data dirs
train_dir = "/hdd1/home/bchaudhary/hyeirn/baseline/baseline/backup/train_new/"
val_dir = "/hdd1/home/bchaudhary/hyeirn/baseline/baseline/backup/val_new/"
test_dir = "/hdd1/home/bchaudhary/hyeirn/baseline/baseline/backup/test_new/"


# GPU config
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0")

# hyperparameters
n_classes = 7
num_epoch = 100
batch_size = 4 
lr = 0.0001

# preprocessing
augmentation_list = [
    normalize,
    to_tensor]


# dataloaders
train_set = ImageFolder(train_dir, transform=augmentation_list)
train_batch = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

test_set = ImageFolder(test_dir, transform=augmentation_list)
test_batch = data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

val_set = ImageFolder(val_dir, transform=augmentation_list)
val_batch = data.DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

# model config
parser = argparse.ArgumentParser(description='Testing')
args = parser.parse_args()

for scale in config_all.sc_ch_dict.keys():
        for size in [600]:
            args.num_classes = 7
            imSz = size
            args.s = scale
            args.channels = 59
            args.model_width = 512
            args.model_height = 512

            model = CNNModel(args)
            break
        break

# move to GPU
if train_on_gpu:
    model = model.to(device)


model_dir = 'highest_test_acc.pt'
try:
    model.load_state_dict(torch.load(model_dir))
    print("model restored!! ")
except:
    print("model not restored!! Training from scratch.......")

# optimizer
optimizer = torch.optim.Adam(model.parameters())


# confusion matrix
confu_matrix = ConfusionMeter(k=n_classes)

# loss function
criterion = nn.CrossEntropyLoss()


# cyclic learning rate
step_size = 300
scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.003,
                         step_size=step_size, mode='exp_range',
                         gamma=0.99994)
    

# acc lists
train_acc_list = []
test_acc_list = []
val_acc_list = []

highest_test_accuracy = torch.FloatTensor([0]).cuda()
highest_val_accuracy = torch.FloatTensor([0]).cuda()

for epoch in range(1, num_epoch+1):
    
    ###################
    # train the model #
    ###################
    model.train()
    top_1_count = torch.FloatTensor([0])
    for train_idx, (img, label) in enumerate(train_batch):
        
        optimizer.zero_grad()
        batch_size = img.shape[0]
        x = Variable(img.type_as(torch.FloatTensor()))
        y = Variable(label)
        if train_on_gpu:
            x = x.to(device)
            y = y.to(device)

        scheduler.batch_step()

        label_out = model(x)
        
        loss = criterion(label_out, y)
        loss.backward()
        optimizer.step()
        
        values, idx = label_out.max(dim=1)

        top_1_count += torch.sum(y == idx).float().cpu().data
    
    
    train_accuracy = 100 * top_1_count.cpu() / ((train_idx + 1) * batch_size)
    print("train accuracy: {}%".format(train_accuracy.numpy()))
    
    train_acc_list.append(train_accuracy.numpy()[0])

    ######################    
    # test the model #
    ######################

    model.eval()
    top_1_count = torch.FloatTensor([0])
    for test_idx, (img, label) in enumerate(test_batch):
        
        x = Variable(img.type_as(torch.FloatTensor()))
        y = Variable(label)

        if train_on_gpu:
            x = x.to(device)
            y = y.to(device)
         
        label_out = model(x)

        values, idx = label_out.max(dim=1)

        top_1_count += torch.sum(y == idx).float().cpu().data

        confu_matrix.add(label_out.data, y.data)

    print("\n----------Confusion Matrix for Test Data----------\n")
    print(confu_matrix.value())
    confu_matrix.reset()

    print((test_idx + 1) * batch_size)
    test_accuracy = 100 * top_1_count.cpu() / ((test_idx + 1) * batch_size)
    print("test accuracy: {}%".format(test_accuracy.numpy()))
    test_acc_list.append(test_accuracy.cpu().numpy()[0])

    ######################    
    # validate the model #
    ######################

    model.eval()
    top_1_count = torch.FloatTensor([0])
    for val_idx, (img, label) in enumerate(val_batch):
    
        x = Variable(img.type_as(torch.FloatTensor()))
        y = Variable(label)
        
        if train_on_gpu:
            x = x.to(device)
            y = y.to(device)
        
        label_out = model(x)

        values, idx = label_out.max(dim=1)
        top_1_count += torch.sum(y == idx).float().cpu().data

    print((test_idx + 1) * batch_size)
    val_accuracy = 100 * top_1_count.cpu() / ((val_idx + 1) * batch_size)
    print("val accuracy: {}%".format(val_accuracy.cpu().numpy()))

    val_acc_list.append(val_accuracy.cpu().numpy()[0])


    with open("train_accuracy_list_dice", "wt") as f:
        f.write(str(train_acc_list))

    with open("test_accuracy_list_dice", "wt") as f:
        f.write(str(test_acc_list))


    if val_accuracy.cpu().numpy() >= 90 or val_accuracy.cpu().numpy() >= highest_val_accuracy.cpu().numpy():
        highest_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'TWT_highest_val_acc.pt')
        print("highest val_acc model saved")

    if test_accuracy.cpu().numpy() >= 90 or test_accuracy.cpu().numpy() >= highest_test_accuracy.cpu().numpy():
        highest_test_accuracy = test_accuracy
        torch.save(model.state_dict(),'highest_test_acc.pt')
        print("highest test_acc model saved")
