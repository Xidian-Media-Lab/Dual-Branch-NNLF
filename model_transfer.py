import torch

device = 'cpu'
netG = torch.load('rootpath/loop_filter_2/checkpoint/AI/UV/REAM/G_epoch_65.pth').to(device)
netG.eval()

# gNet = torch.jit.trace(netG, torch.rand(1, 4, 144, 144).to(device))
# gNet.save("pt/AI/filter_Y.pt")

gNet = torch.jit.trace(netG, torch.rand(1, 10, 72, 72).to(device))
gNet.save("./pt/AI/filter_UV.pt")

