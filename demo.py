import  pandas as pd
import torch

# with open("dataset//0.txt","r") as f:
#     data=f.readline()
#     a=[]
#     for i in data:
#         a.append(int(i))
#     a=torch.tensor(a)
#     print(a)
frequency=pd.read_csv("filelabel.csv")
print(torch.tensor(frequency.iloc[0]))