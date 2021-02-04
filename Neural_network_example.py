import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create Fully connected network

class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__() ## Call nn.module's initialization

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NN(784,10)
x = torch.randn(64,784) ## 64 Mini batch 
print(model(x).shape) ## Should be [64,10]

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

transform = transforms.Compose([transforms.ToTensor(),
                                    ])

# Load Data
train_data = datasets.MNIST(root = '/home/sungsu21/Project/data',train = True, transform = transform, download = True)
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_data = datasets.MNIST(root = '/home/sungsu21/Project/data',train = False, transform = transform, download = True)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)

# Initialize network

model = NN(input_size = input_size, num_classes = num_classes).to(device)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)


for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):

        data = data.to(device= device)
        targets = targets.to(device = device)

        #print(data.shape) # [64,1,28,28] 

        data = data.reshape(data.shape[0],-1) ## Make data in single dimension

        # forward

        scores = model(data)
        loss = criterion(scores,targets)

        # backward

        optimizer.zero_grad() # initialize gradient 
        loss.backward()

        # gradient descent or Adam step

        optimizer.step()


# Check Accuracy on training & test to see how good our model

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
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            # scores : 64 X 10-->> Find maximum probability class
            # scores.max(1) --> (values, indices)
            _, predictions = scores.max(1) # We are interested in indices 
            if num_correct == 0:
                print(scores.max(1),y)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy  {float(num_correct)/float(num_samples)*100:.2f}')
        model.train()
    
Check_Accuracy(train_loader,model)
Check_Accuracy(test_loader,model)