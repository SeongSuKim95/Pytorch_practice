import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
# Methods for dealing with imbalanced datasets:
# 1. Oversampling --> Perform different data augmentations in single data
# 2. Class weighting

# Class weighting
loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([1,50])) # Weight X50 on elkhound 


def get_loader(root_dir,batch_size):

    my_transforms = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ]
    )

    dataset = datasets.ImageFolder(root = root_dir, transform = my_transforms) # ImageFolder

    ##class_weights = [1,50] --> Just relative weight difference 
    
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))
    
    
    sample_weights = [0] * len(dataset) # Initialize

    for idx, (data,label) in enumerate(dataset):

        class_weight = class_weights[label] 
        sample_weights[idx] = class_weight
    
    sampler = WeightedRandomSampler(sample_weights, num_samples = len(sample_weights), replacement = True)
    # replacement = True for oversampling
    
    loader = DataLoader(dataset,batch_size = batch_size, sampler = sampler)

    return loader


def main():

    loader = get_loader(root_dir = "Imbalance_dog_data", batch_size = 8)
    

    num_retrievers = 0
    num_elkounds = 0
    for epoch in range(10):
         
        for data, labels in loader:
            num_retrievers += torch.sum(labels == 0)
            num_elkounds += torch.sum(labels == 1)

        print(num_retrievers, num_elkounds)
            # print(labels)
            # Balanced labels
            # tensor([1, 1, 1, 1, 0, 0, 0, 0])
            # tensor([1, 0, 1, 0, 0, 0, 0, 1])
            # tensor([1, 0, 1, 1, 0, 1, 0, 0])
            # tensor([0, 0, 1, 1, 1, 0, 0, 0])
            # tensor([1, 1, 0, 1, 0, 0, 0, 0])
            # tensor([0, 1, 0, 1, 1, 1, 1, 1])
            # tensor([0, 0, 0])


if __name__ == "__main__":
    main()

