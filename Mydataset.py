import os

import torch
from torch.utils.data import Dataset
import pandas as pd
class myDataset(Dataset):
    def __init__(self,dir,frequencydir ):
        self.data = os.listdir(dir)
        self.frequencydir=os.listdir(frequencydir)

    def __len__(self):
        return len(self.csv_data)

    def __gettime__(self,idx):
        data = (self.data[idx],self.txt_data[idx])
        return data
import torch.nn as nn
print(callable(nn.Sequential))