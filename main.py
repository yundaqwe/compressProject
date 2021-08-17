import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Mydataset import myDataset
from generator import GeneratorA
from generator2 import GeneratorB

batch_size = 64
z_dim = 100
z_sample = Variable(torch.randn(z_dim))
print(z_sample)
lr = 1e-4

""" Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
n_epoch = 1 # 50
n_critic = 1 # 5
clip_value = 0.01

log_dir = 'logs'
ckpt_dir =  'checkpoints'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Model
GA = GeneratorA(256, 256)
GB = GeneratorB(1024 * 100,256)
GA.train()
GB.train()

# Loss
criterion = nn.BCELoss()

""" Medium: Use RMSprop for WGAN. """
# Optimizer
opt_D = torch.optim.Adam(GB.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(GA.parameters(), lr=lr, betas=(0.5, 0.999))



# DataLoader
dataset=myDataset("dataset","filelabel.csv")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)#TODO
z = Variable(torch.randn( z_dim))
a = []
with open("dataset//0.txt", "r") as f:
    data = f.readline()

    for i in data:
        a.append(int(i))
    a = torch.tensor(a)
f_imgs = GB(a)
print(f_imgs)
