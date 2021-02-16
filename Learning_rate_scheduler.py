import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import os

## device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICE"] = "1"


## Hyperparams

num_channels = 3
num_classes = 1000
learning_rate = 1e-3
batch_size = 16
num_epochs = 10

## Pretrained model

model = torchvision.models.googlenet(pretrained = True)

for param in model.parameters():
    param.requires_grad = False # Freeze layers

model.fc = nn.Linear(1024, num_classes)
model.to(device)

train_dataset = datasets.CIFAR10(root = "/home/sungsu21/Project/data/", train = True, transform = transforms.ToTensor())
test_dataset = datasets.CIFAR10(root = "/home/sungsu21/Project/data/", train = False, transform = transforms.ToTensor())

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

## Criterion

criterion = nn.CrossEntropyLoss()

## Optimizer

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

## Scheduler

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, verbose = True) # If loss doesn't decrease for specific number of epoch(in this case, 5)

for epoch in range(num_epochs) :
    losses = []
    for idx, (images, labels) in enumerate(train_loader):

        images = images.to(device=device)
        labels = labels.to(device=device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        losses.append(loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    
    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)
    print(f'Cost ae epoch {epoch} is {mean_loss}')