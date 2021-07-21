import torch
import numpy as np

from torch.utils.data import Dataset, ConcatDataset, Sampler, RandomSampler, DataLoader
from torch.utils.data.sampler import BatchSampler


class MapDataset(Dataset):

    def __len__(self):

        return 12

    def __getitem__(self, index) :

        return {"Input" : torch.tensor([index, 2* index , 3* index], dtype= torch.float32),
                "label" : torch.tensor(index, dtype = torch.float32)}


class VarMapDataset(Dataset):

    def __len__(self):

        return 10

    def __getitem__(self, index) :

        return {"Input" : torch.tensor([index]*(index+1), dtype= torch.float32),
                "label" : torch.tensor(index, dtype = torch.float32)}

    
# map_dataset = MapDataset()
# point_sampler = RandomSampler(map_dataset)
# batch_sampler = BatchSampler(point_sampler,3,False)

# dataloader = torch.utils.data.DataLoader(map_dataset, batch_sampler = batch_sampler)

# for data in dataloader:

#     print(data)
#     print("-------------")


var_map_dataset = VarMapDataset()

dataloader = torch.utils.data.DataLoader(var_map_dataset)
# dataloader = torch.utils.data.DataLoader(var_map_dataset,batch_size =2) : data의 크기가 달라 batch로 묶을 수 없음

def make_batch(samples):

    inputs = [sample["Input"] for sample in samples]
    labels = [sample["label"] for sample in samples]

    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs,batch_first = True)
    
    return {'input': padded_inputs.contiguous(),
            'label': torch.stack(labels).contiguous()}

dataloader = torch.utils.data.DataLoader(var_map_dataset,
                                         batch_size=3,
                                         collate_fn=make_batch)
for data in dataloader:
    print(data['input'], data['label'])

