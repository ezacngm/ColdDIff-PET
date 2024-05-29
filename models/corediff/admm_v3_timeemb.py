import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import ssim


def normalize_(img, min_val=None, max_val=None):
    min_val = img.min().item() if min_val is None else min_val
    max_val = img.max().item() if max_val is None else max_val
    # Check if max_val equals min_val to prevent division by zero
    if max_val - min_val == 0:
        img_normalized = img.clone().fill_(0)  # Or any default value you deem appropriate
    else:
        # Normalize PET images to a [0, 1] range
        img_normalized = (img - min_val) / (max_val - min_val)
    # Clip values just in case to ensure they remain within [0, 1]
    img_normalized = img_normalized.clip(0, 1)
    return img_normalized

class XUpdateBlock(nn.Module):
    def __init__(self,pbeam):
        super(XUpdateBlock, self).__init__()
        self.pbeam= pbeam
        self.conv_i2o64_1 = nn.Conv2d(2, 64, 3, padding=1)
        self.conv_i64o64_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_i64o1_1 = nn.Conv2d(64, 1, 3, padding=1)

        self.conv_i3o64_2 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv_i64o64_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_i64o1_2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, ins):
        #
        x = ins[0]
        y = ins[1]  # torch.unsqueeze(ins[1],1)
        z = ins[2]
        b = ins[3]
        # -----
        x_bp = self.pbeam.forward(x)
        # x_bp = self.pbeam.filter_sinogram(x_bp)

        #取y的两个channel

        # sino = y[:,0:1,:,:]
        # concat = torch.cat([x_bp, sino], 1)

        concat = torch.cat([self.pbeam.forward(x), y], 1)
        Ax = self.conv_i2o64_1(concat)
        Ax = self.conv_i64o1_1(Ax)
        Ayh = self.pbeam.backward(Ax)/180

        # Ayh = self.pbeam.backward(Ax)/360
        # Ayh = normalize_(Ayh)

        # -----
        x1 = F.leaky_relu(self.conv_i3o64_2(torch.cat([x, z - b, Ayh], 1)))
        x1 = self.conv_i64o64_2(x1)
        x1 = self.conv_i64o1_2(x1)
        #
        return (x1, y, z, b)


class ZUpdateBlock(nn.Module):
    def __init__(self):
        super(ZUpdateBlock, self).__init__()

        self.conv_i1o64 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv_i64o64 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_i64o1 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, ins):
        #
        x = ins[0]
        y = ins[1]  # torch.unsqueeze(ins[1],1)
        z = ins[2]
        b = ins[3]
        # -----

        # 取y的两个channel

        # recon = y[:, 1:2, :, :]
        # z1 = F.leaky_relu(self.conv_i1o64(torch.cat([x + b, recon], dim=1)))

        z1 = F.leaky_relu(self.conv_i1o64(x + b))
        z1 = self.conv_i64o64(z1)
        # z1 =  (self.conv_i1o32(x+b))
        # z1 = (self.conv_i32o32(z1))
        z1 = self.conv_i64o1(z1)
        # -----
        #
        return (x, y, z1, b)


class InterBlock(nn.Module):
    def __init__(self, pbeam):
        super(InterBlock, self).__init__()
        self.eta = torch.nn.Parameter(torch.tensor(0.001), requires_grad=True)

        self.layers_up_x = []
        for i in range(2):  # 1子迭代
            self.layers_up_x.append(XUpdateBlock(pbeam))
        self.net_x = nn.Sequential(*self.layers_up_x)

        self.layers_up_z = []
        for i in range(2):  # 1子迭代
            self.layers_up_z.append(ZUpdateBlock())
        self.net_z = nn.Sequential(*self.layers_up_z)

    def forward_inloop(self, ins):
        #
        x = ins[0]
        y = ins[1]  # torch.unsqueeze(ins[1],1)
        z = ins[2]
        b = ins[3]
        # -----x
        [x1, y, z, b] = self.net_x([x, y, z, b])
        # x1 = nn.ReLU()(x1)
        # ------z
        [x1, y, z1, b] = self.net_z([x1, y, z, b])
        # -------b
        b1 = b + self.eta * (x1 - z1)
        #
        return (x1, y, z1, b1)

    def forward(self, ins):
        for i in range(15):
            ins = self.forward_inloop(ins)
        return ins[0], ins[1], ins[2], ins[3]


class ADMM_Net3(nn.Module):
    def __init__(self, n_iter, pbeam):
        super(ADMM_Net3, self).__init__()
        self.n_iter = 1
        self.layers = []
        self.batch_sz=1
        self.pbeam = pbeam
        self.count = 20e4

        for i in range(1):
            self.layers.append(InterBlock(pbeam))
        self.net = nn.Sequential(*self.layers)
        self.x0 = torch.zeros((self.batch_sz, 1, 64, 64), dtype=torch.float32).cuda()
        self.z0 = torch.zeros((self.batch_sz, 1, 64, 64), dtype=torch.float32).cuda()
        self.b0 = torch.zeros((self.batch_sz, 1, 64, 64), dtype=torch.float32).cuda()
        # self.b0 = torch.nn.Parameter(torch.tensor(0.003),requires_grad=True).cuda()
        # self.b0 = torch.tensor(0.0003,requires_grad=False).cuda()

    def addnoise(self,proj,countnumber):
        mul_factor = torch.ones_like(proj)
        mul_factor = mul_factor + (torch.rand_like(mul_factor) * 0.2 - 0.1)
        noise = torch.ones_like(proj) * torch.mean(mul_factor * proj, dim=(-1, -2), keepdims=True) * 0.2
        sinogram = mul_factor * proj + noise
        cs = countnumber / (1e-9 + torch.sum(sinogram, dim=(-1, -2), keepdim=True))
        sinogram = sinogram * cs
        mul_factor = mul_factor * cs
        noise = noise * cs
        noisy_sino = torch.poisson(sinogram)
        sino = nn.ReLU()((noisy_sino - noise) / mul_factor)
        return sino

    def forward(self, x, low_dose):
        '''
        :param x: 扩散模型重建出的图像，应当作为先验项输入
        :param low_dose: 基础重建图像，应该用他反变换的sinogram作为admm保真项输入


        所以输入y应该有两个channel，第一个channel是 inverse randon(low_dose),第二个channel是 重建图像x
        '''


        # sino_lowdose = self.pbeam.forward(low_dose)
        # final_proj = torch.cat((sino_lowdose, x),dim=1)
        proj_lowdose = self.pbeam.forward(low_dose)

        proj_diff = self.pbeam.forward(x)

        # noisy_proj_diff = self.addnoise(proj_diff,5e4)

        proj_combined = torch.cat((proj_lowdose,proj_diff),dim=1)

        [xst, yst, zst, bst] = self.net((self.x0, proj_diff, self.z0, self.b0))
        # print("admmout",xst.min().item(),xst.max().item())
        # xst = normalize_(xst)
        # print(xst.min().item(), xst.max().item())
        # xst = torch.clamp(xst, min=0.,max=1.0)
        proj_out = self.pbeam.forward(xst)


        return xst, x, proj_out
def admm():
    from torch_radon import ParallelBeam
    pbeam = ParallelBeam(128,np.linspace(0, np.pi, 128, endpoint=False))
    net = ADMM_Net3(15,pbeam)
    return net
