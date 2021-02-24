from __future__ import print_function, division
from dataset import Graph_Dataset
import torch
import torchvision.models as models
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import os

device = torch.device("cuda")
dataset = Graph_Dataset()

# 0.8-0.2 train test
train_max = 0.8 * len(dataset)
test_min = train_max + 1

model = models.resnet18(num_classes=3)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)


def train_model(model, critirion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        print("-" * 10)

        for phase in ['train', 'test']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0 

            if phase == 'train':
                for index in tqdm(range(int(train_max))):
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        inputs, labels = dataset[index]
                        outputs = model(torch.unsqueeze(inputs, 0))
                        # _, preds = torch.max(outputs, 1)
                        loss = critirion(outputs, torch.tensor([labels]))
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += 1
                scheduler.step()
                        
            if phase == 'test':
                for index in tqdm(range(int(test_min), int(len(dataset)))):
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(False):
                        inputs, labels = dataset[index]
                        outputs = model(torch.unsqueeze(inputs, 0))
                        loss = critirion(outputs, torch.tensor([labels]))
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += 1
    
            epoch_loss = running_loss / train_max
            epoch_acc  = running_corrects / train_max

            print(f"phase Loss {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_params = copy.deepcopy(model.state_dict())
    
    timep = time.time() - since
    print('Done')
    model.load_state_dict(best_model_params)
    torch.save(model, "./model")
    return model

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
