import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToPILImage

from Dataset import CatsAndDogsDataset

## Transforms

my_transform = transforms.Compose(
            [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5),
            transforms.Resize((256,256)),
            transforms.RandomRotation(degrees = 45),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            #After ToTensor
            #Normalize per channels
            transforms.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]) # (value - mean) / std Note this does nothing
            ]
                              )

Dataset = CatsAndDogsDataset("/home/sungsu21/Project/Pytorch_practice/practice_data",
                             "/home/sungsu21/Project/Pytorch_practice/train_csv.csv",transform = my_transform)


train_loader = DataLoader(dataset = Dataset, shuffle = True, batch_size = 1)
img_num = 0
for _ in range(10):
    for img, label in Dataset:
        save_image(img, 'img'+str(img_num)+'.png')
        img_num += 1
        print(img.shape)
