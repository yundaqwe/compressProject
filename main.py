import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Mydataset import myDataset
from generator import GeneratorA
from generator2 import GeneratorB
from qqdm  import qqdm
batch_size = 64
z_dim = 100
z_sample = Variable(torch.randn(z_dim))
print(z_sample)
lr = 1e-4

""" Medium: WGAN, 50 epoch, n_critic=5, clip_value=0.01 """
n_epoch = 500 # 50
n_critic = 5 # 5
clip_value = 0.01

log_dir = 'logs'
ckpt_dir =  'checkpoints'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Model
GA = GeneratorA(256, 256).cuda()
GB = GeneratorB(1024 * 100,256).cuda()
GA.train()
GB.train()

# Loss
criterion = nn.MSELoss()

""" Medium: Use RMSprop for WGAN. """
# Optimizer
opt_D = torch.optim.Adam(GB.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(GA.parameters(), lr=lr, betas=(0.5, 0.999))



# DataLoader
dataset=myDataset("dataset","filelabel.csv")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)#TODO
z = Variable(torch.randn( z_dim))
a = []
steps = 0
for e, epoch in enumerate(range(n_epoch)):
    progress_bar = qqdm(dataloader)
    for i, data in enumerate(progress_bar):
        file = data[0].float()
        file = file.cuda()

        bs = file.size(0)

        # ============================================
        #  Train GB
        # ============================================
        # z = Variable(torch.randn(bs, z_dim)).cuda()
        r_file = Variable(file).cuda()
        z_code = GB(r_file)
        f_file=GA(z_code.detach(),data[1].float().cuda())

         # Compute the loss for the discriminator.
        loss = criterion(f_file, r_file).cuda()
        # f_loss = criterion(f_logit, f_label)
        loss_D = loss / 2

        # WGAN Loss
        # loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))

        # Model backwarding
        GA.zero_grad()
        GB.zero_grad()
        loss_D.backward()

        # Update the discriminator.
        opt_D.step()
        opt_G.step()

        steps += 1

        # Set the info of the progress bar
        #   Note that the value of the GAN loss is not directly related to
        #   the quality of the generated images.
        progress_bar.set_infos({
            'Loss': loss_D,
          # 'Epoch': e + 1,
            'Step': steps,
        })

    # G.eval()
    # f_imgs_sample = (G(z_sample).data + 1) / 2.0
    # filename = os.path.join(log_dir, f'Epoch_{epoch + 1:03d}.jpg')
    # torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    # print(f' | Save some samples to {filename}.')
    #
    # # Show generated images in the jupyter notebook.
    # grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(grid_img.permute(1, 2, 0))
    # plt.show()
    GA.train()
    GB.train()

    if (e + 1) % 5 == 0 or e == 0:
        # Save the checkpoints.
        torch.save(GA.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
        torch.save(GB.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

# with open("dataset//0.txt", "r") as f:
#     data = f.readline()
#
#     for i in data:
#         a.append(int(i))
#     a = torch.unsqueeze(torch.tensor(a).float(),1).view(1,-1)
# print(a.shape)
# GB.eval()
# f_imgs = GB(a)
# print(f_imgs)
