import torch
from torch import nn


class GeneratorB(nn.Module):
    def __init__(self, filesize,codesize):
        super().__init__()
        self.filesize = filesize
        self.codesize = codesize
        net = []
        channels_in = [self.filesize ,  2048,1024]
        # channels_in = [self.noise + self.frequency, 512, 256, 128, 64]
        channels_out = [2048, 1024,self.codesize ]
        active = ["R", "R",  "R"]

        for i in range(len(channels_in)):
            net.append(nn.Linear(in_features=channels_in[i], out_features=channels_out[i],
                                         bias=True))
            if active[i] == "R":
                net.append(nn.BatchNorm1d(num_features=channels_out[i]))
                net.append(nn.ReLU())
            elif active[i] == "tanh":
                net.append(nn.Tanh())

        self.generator = nn.Sequential(*net)

    def forward(self, x):
        out = self.generator(x)
        return out
# # Training hyperparameters
# batch_size = 64
# z_dim = 100
# lr = 1e-4
# n_epoch = 1 # 50
# n_critic = 1 # 5
# # clip_value = 0.01
# # Model
# G = Generator(256,256)
# for module in G.modules():
#     print(module)
# # G.train()
# # Loss
# criterion = nn.CrossEntropyLoss()
# # Optimizer
# opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))