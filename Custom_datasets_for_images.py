import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataloader
from Custom_datasets import CatsAndDogsDataset

# Set device
device = torch.device('cuda' if torch.cude.is_available() else 'cpu')

# Hyperparams

in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 5

# Load data

dataset = CatsAndDogsDataset(csv_file = "/home/sungsu21/Project/data/dogs_cats_data/sampleSubmission.csv", root_dir = 'cats_dogs_resized', 
                             transform = transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset,[20000,5000])

train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

# Pretrained model

model = torchvision.models.googlenet(pretrained = True)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):

    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):

            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores,targets)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')


