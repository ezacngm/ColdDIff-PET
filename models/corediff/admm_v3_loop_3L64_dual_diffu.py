import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import ssim
from torch_radon import ParallelBeam
from .unet import UNet

# Unet initialisation


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

    def radontrans_batch(self,x,direction:str):
        processed_images = []
        for img in x:
            if direction == "forward":
                img_processed = self.pbeam.forward(img.unsqueeze(0))  # 加一个 batch 维
            if direction == "backward":
                img_processed = self.pbeam.backward(img.unsqueeze(0))/180 # 加一个 batch 维
            processed_images.append(img_processed)
        batch_x_transformed = torch.cat(processed_images, 0)
        return batch_x_transformed

    def forward(self, ins):
        #
        x = ins[0]
        y = ins[1]  # torch.unsqueeze(ins[1],1)
        z = ins[2]
        b = ins[3]
        # -----
        Ax = self.conv_i2o64_1(torch.cat([self.radontrans_batch(x, direction="forward"), y], 1))
        Ax = self.conv_i64o64_1(Ax)
        Ax = self.conv_i64o1_1(Ax)
        Ayh = self.pbeam.backward(Ax) / 180  #注意这里又将sinogram反变换成了图像
        # -----
        x1 = F.leaky_relu(self.conv_i3o64_2(torch.cat([x, z - b, self.radontrans_batch(Ax,direction="backward")], 1)))
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
        z1 = F.leaky_relu(self.conv_i1o64(x + b))
        z1 = self.conv_i64o64(z1)
        # z1 =  (self.conv_i1o32(x+b))
        # z1 = (self.conv_i32o32(z1))
        z1 = self.conv_i64o1(z1)
        # -----
        #
        return (x, y, z1, b)


class InterBlock(nn.Module):
    def __init__(self, pbeam, noiser=None, noisy=True):
        super(InterBlock, self).__init__()
        self.eta = torch.nn.Parameter(torch.tensor(0.001), requires_grad=True)
        self.noisy = noisy
        self.noiser = noiser

        self.layers_up_x = []
        for i in range(2):  # 1子迭代
            self.layers_up_x.append(XUpdateBlock(pbeam))
        self.net_x = nn.Sequential(*self.layers_up_x)

        self.layers_up_z = []
        for i in range(2):  # 1子迭代
            self.layers_up_z.append(ZUpdateBlock())
        self.net_z = nn.Sequential(*self.layers_up_z)

    def forward_inloop(self, ins, i):
        #d

        x = ins[0]
        y = ins[1]  # torch.unsqueeze(ins[1],1)
        z = ins[2]
        b = ins[3]
        device = x.device
        t = torch.tensor([15-i], device=device)
        # -----x
        [x1, y, z, b] = self.net_x([x, y, z, b])
        # x1 = nn.ReLU()(x1)
        # ------z
        [x1, y, z1, b] = self.net_z([x1, y, z, b])
        # -------b
        b1 = b + self.eta * (x1 - z1)
        # ---- Unet x update
        if not self.noisy:
            return (x1, y, z1, b1)
        noise_upre = self.noiser(x1, time=t)
        x1 = x1 - noise_upre
        # x1 = nn.ReLU()(x1)
        return (x1, y, z1, b1, noise_upre)


    def forward(self, ins):

        if not self.noisy:
            x_list = []
            for i in range(15):
                ins = self.forward_inloop(ins, i)
                # 第一次输入是ins=【x0，img的sino，z0，b0】，接下来进入xzu更新网络(叫一个forward_inloop)，
                # 输出一个ins[x1,image的sino,z1,b1]，再马上进入下一个forwad_inloop, 注意此时image的sino是不变的，
                # 所以 接下来ADMM_Net3中的加噪只用加一次
                x_list.append(ins[0])
            return ins[0], ins[1], ins[2], ins[3], x_list
        x_list = []
        noise_upre_list = []
        for i in range(15):
            ins = self.forward_inloop(ins, i)
            # 第一次输入是ins=【x0，img的sino，z0，b0】，接下来进入xzu更新网络(叫一个forward_inloop)，
            # 输出一个ins[x1,image的sino,z1,b1]，再马上进入下一个forwad_inloop, 注意此时image的sino是不变的，
            # 所以 接下来ADMM_Net3中的加噪只用加一次
            x_list.append(ins[0])
            noise_upre_list.append(ins[-1])
        return ins[0], ins[1], ins[2], ins[3], x_list, noise_upre_list


class ADMM_Net3(nn.Module):
    def __init__(self, pbeam, noiser=None, noisy=True):
        super(ADMM_Net3, self).__init__()
        self.noisy = noisy
        self.noiser = noiser
        self.n_iter = 1 #todo: 问这个为啥是1 不是10
        self.batch_sz = 1
        self.pbeam = pbeam
        self.count = 20e4
        self.layers = []
        self.layers.append(InterBlock(pbeam, self.noiser, noisy=self.noisy))
        self.net = nn.Sequential(*self.layers) # unpacking self.layers[] list

        # initialisation of the first input, gradually changed to PET image
        self.x0 = torch.zeros((self.batch_sz, 1, 128, 128), dtype=torch.float32).cuda()
        self.z0 = torch.zeros((self.batch_sz, 1, 128, 128), dtype=torch.float32).cuda()
        self.b0 = torch.zeros((self.batch_sz, 1, 128, 128), dtype=torch.float32).cuda()
        # self.x0 = torch.randn((self.batch_sz, 1, 128, 128), dtype=torch.float32).cuda()
        # self.z0 = torch.randn((self.batch_sz, 1, 128, 128), dtype=torch.float32).cuda()
        # self.b0 = torch.randn((self.batch_sz, 1, 128, 128), dtype=torch.float32).cuda()
        # self.b0 = torch.nn.Parameter(torch.tensor(0.003),requires_grad=True).cuda()
        # self.b0 = torch.tensor(0.0003,requires_grad=False).cuda()

    def radontrans_batch(self,x,direction:str):
        processed_images = []
        for img in x:
            if direction == "forward":
                img_processed = self.pbeam.forward(img.unsqueeze(0))  # 加一个 batch 维
            if direction == "backward":
                img_processed = self.pbeam.backward(img.unsqueeze(0))/180 # 加一个 batch 维
            processed_images.append(img_processed)
        batch_x_transformed = torch.cat(processed_images, 0)
        return batch_x_transformed

    def forward(self, x):
        # 输入x为已经有的高剂量PEimg，要把x先投影成sinogram，再在sinogram上加噪音，加噪的sinogram作为网络输入，再看一下x更新网络的注释，训练出预测的高剂量PETimg‘， 和PETimg 做比较
        proj = self.radontrans_batch(x, direction="forward")  # 投影到sinogram
        if not self.noisy:
            [xst, yst, zst, bst, x_list] = self.net((self.x0, proj, self.z0, self.b0))
            return xst, x_list
        mul_factor = torch.ones_like(proj)
        mul_factor = mul_factor + (torch.rand_like(mul_factor) * 0.2 - 0.1)
        noise = torch.ones_like(proj) * torch.mean(mul_factor * proj, dim=(-1, -2), keepdims=True) * 0.2
        sinogram = mul_factor * proj + noise
        cs = self.count / (1e-9 + torch.sum(sinogram, dim=(-1, -2), keepdim=True))
        sinogram = sinogram * cs
        mul_factor = mul_factor * cs
        noise = noise * cs
        x = torch.poisson(sinogram)
        sino = nn.ReLU()((x - noise) / mul_factor)

        [xst, yst, zst, bst, x_list, noise_upre_list] = self.net((self.x0, sino, self.z0, self.b0)) #todo: 加x_list

        proj_out = self.radontrans_batch(xst,direction="forward")
        sinogram_out = mul_factor * proj_out + noise

        return xst, sinogram_out-sinogram, x_list, noise_upre_list

#todo: 创建一个新的 Dual-Admmnet
class Dual_ADMM_diffu_Net3(nn.Module):
    def __init__(self, pbeam, noiser):
        super(Dual_ADMM_diffu_Net3, self).__init__()
        self.noiser = noiser
        self.net_cleansino = ADMM_Net3(pbeam,noisy=False)
        self.net_noisysino = ADMM_Net3(pbeam,noiser=self.noiser)

    def forward(self, x):
        cleanx, cleanx_list =  self.net_cleansino(x)
        noisyx, diff, noisyx_list , noise_upre_list = self.net_noisysino(x)
        return cleanx, cleanx_list, noisyx, noisyx_list, diff, noise_upre_list# others

#todo: 配置网络的时候回传
def admm():
    pbeam = ParallelBeam(128,np.linspace(0, np.pi, 128, endpoint=False))
    # noiser = UNet(
    #     img_channels = 1,
    #     base_channels = 32,
    #     channel_mults=(1, 2, 4, 8, 16),
    #     time_emb_dim=128,
    #     attention_resolutions=(4,),
    #     num_classes=None,
    #     initial_pad=0,
    # )
    noiser = UNet(
        img_channels = 1,
        base_channels = 64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=128,
        attention_resolutions=(1,),
        num_classes=None,
        initial_pad=0,
    )
    final_net = Dual_ADMM_diffu_Net3(pbeam,noiser)
    return final_net
