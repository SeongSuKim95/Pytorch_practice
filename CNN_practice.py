import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import dataloader
from torch.utils.data.dataloader import DataLoader


import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):

    def __init__(self,input_size, num_classes):
        super(NN,self).__init__() 

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class CNN(nn.Module):

    def __init__(self,in_channels = 1, num_classes = 10): # input_size -> in_channels 
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)) 
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.fc1 = nn.Linear( 16 * 7 * 7, num_classes) # 28 / 4 -> Maxpool , Stride
    
    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) # x.shape[0] = batch size

        x = self.fc1(x)

        return x
         
        # n_in = number of input features
        # k  = convolution kernel size
        # p  = convolution padding size
        # s  = convolution stride size

        # n_out = number of output features = [n_in + 2p -k ]/s +1

# Device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparams
# in_channels, num_classes, batch_size, epochs, learning_rate
in_channels = 1 # input_size -> in_channels
num_classes = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Data

train_dataset = datasets.MNIST(root = '/home/sungsu21/Project/data', train = True, transform = transforms.ToTensor(),  download = False)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle =True)

test_dataset = datasets.MNIST(root = '/home/sungsu21/Project/data', train = False, transform = transforms.ToTensor(),  download = False)
test_loader =  DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# Initializing network

model = CNN(in_channels = in_channels , num_classes = num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):

        # Get data to cuda
        data = data.to(device = device)
        targets = targets.to(device = device)

        # forward
        scores = model(data)
        print(scores.shape, targets.shape)

        loss = criterion(scores,targets)

        # backward

        optimizer.zero_grad()
        loss.backward()

        # gradiennt descent or adam step
        optimizer.step()

def Check_Accuracy(loader,model):
    if loader.dataset.train:
        print("Checking accuaracy on train data")
    else:
        print("Checking accuaracy on test data")
    
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader :
            x = x.to(device = device)
            y = y.to(device = device)

            scores = model(x)
            # scores : 64 X 10-->> Find maximum probability class
            # scores.max(1) --> (values, indices)
            _, predictions = scores.max(1) # We are interested in indices 

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy  {float(num_correct)/float(num_samples)*100:.2f}')
        model.train()

Check_Accuracy(train_loader,model)
Check_Accuracy(test_loader,model)