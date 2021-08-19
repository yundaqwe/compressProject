import os

import torch
from torch.utils.data import Dataset
import pandas as pd
class myDataset(Dataset):
    def __init__(self,dir,frequencycsv ):
        self.data = os.listdir(dir)
        for i in range(len(self.data)):
            self.data[i]=os.path.join(dir,self.data[i])
        self.frequency=pd.read_csv(frequencycsv,index_col=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        a = []
        with open(self.data[idx], "r") as f:
            data = f.readline()

            for i in data:
                a.append(int(i))
            a = torch.tensor(a)
        data = (a,torch.tensor(self.frequency.iloc[idx]))
        return data


