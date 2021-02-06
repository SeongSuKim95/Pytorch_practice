import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class CatsAndDogsDataset(Dataset):
    
    def __init__(self,csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.path_list = ["image1", "image2", "image3"]
        self.gt_list = ["1", "2", "1"]

    def __len__(self):
        return len(self.annotations) # 25000
    
    def __getitem__(self, index):
        image = cv2.imread(self.path_list[index])
        y_label = self.gt_list[index]


        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        # pandas.DataFrame.iloc
        # .iloc[] is primarily integer position based(from 0 to length -1 of the axis),but may also be used with a boolean array
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:

            image = self.transform(image)
        
        return (image, y_label)
        