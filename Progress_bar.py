import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm 

class CNN(nn.Module):

    def __init__(self,in_channels, num_classes):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
        self.pool  = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.fc = nn.Linear(16 * 8 * 8 , num_classes)

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)

        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparams

in_channels = 3
num_classes = 10
num_epochs = 5
learning_rate = 1e-3
batch_size = 16

transform = transforms.Compose(
    [
    transforms.ToTensor()
    ]
)

train_dataset = datasets.CIFAR10(root = "/home/sungsu21/Project/data/", train = True, download = False, transform = transform)
test_dataset = datasets.CIFAR10(root = "/home/sungsu21/Project/data/", train = False, download = False, transform = transform)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False)

model = CNN(in_channels = 3, num_classes = 10).to(device)

# Optimizer

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Loss function

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    
    loop = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)
    
    for batch_idx, (images, labels) in loop:
        
        images = images.to(device = device)
        labels = labels.to(device = device)

        outputs = model(images)

        loss = criterion(outputs,labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # update progress bar

        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss = loss.item(), acc = torch.rand(1))
# tqdm(enumerate(train_loader), leave =False, total = len(train_loader)):