import os

import torch
from torch.utils.data import Dataset
import pandas as pd
class myDataset(Dataset):
    def __init__(self,dir,frequencycsv ):
        self.data = os.listdir(dir)
        for i in range(len(self.data)):
            self.data[i]=os.path.join(dir,self.data[i])
        self.frequency=pd.read_csv(frequencycsv)

    def __len__(self):
        return len(self.data)

    def __gettime__(self,idx):
        data = (self.data[idx],self.frequency.iloc[[idx]])
        return data


