import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from Custom_datasets import CatsAndDogsDataset

# Load Data

my_transforms = transforms.ToTensor()

