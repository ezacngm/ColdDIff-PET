# CoreDiff trainer class.
# This part builds heavily on https://github.com/arpitbansal297/Cold-Diffusion-Models.
import torch
from torch import nn
import numpy as np
import torchvision
import math
from .admm_v3 import ADMM_Net3
from torch_radon import ParallelBeam
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def normalize_(img, min_val=None, max_val=None):
    min_val = img.min().item()
    max_val = img.max().item()
    # Normalize PET images to a [0, 1] range
    img_normalized = (img - min_val) / (max_val - min_val)
    # Clip values just in case to ensure they remain within [0, 1]
    img_normalized = img_normalized.clip(0, 1)
    # img = transforms.RandomRotation(degrees=(-15, 15))(img)  # todo: 看torch文档 要求输入shape是 BCWH
    # img = transforms.RandomHorizontalFlip(p=0.5)(img)
    return img_normalized

def linear_alpha_schedule(timesteps):
    steps = timesteps
    alphas_cumprod = 1 - torch.linspace(0, steps, steps) / timesteps
    return torch.clip(alphas_cumprod, 0, 0.999)


class Diffusion(nn.Module):
    def __init__(self,
        denoise_fn = None,
        admmnet = None,
        image_size = 128,
        channels = 1,
        timesteps = 10,
        context=False,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.admmnet = admmnet
        self.num_timesteps = int(timesteps)
        self.context = context
        self.lmd_interval = 6
        alphas_cumprod = linear_alpha_schedule(timesteps)
        lmd = torch.pow(8,(alphas_cumprod-1/2)*self.lmd_interval)

        self.register_buffer('lmd', lmd)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('one_minus_alphas_cumprod', 1. - alphas_cumprod)

    #  mean-preserving degradation operator
    def q_sample(self, x_start, x_end, t):

        # xt = (extract(self.alphas_cumprod, t, x_start.shape) * torch.poisson(x_start*extract(self.lmd, t, x_start.shape))/extract(self.lmd, t, x_start.shape) +
        #       extract(self.one_minus_alphas_cumprod, t, x_start.shape) * extract(self.lmd, t, x_start.shape)*torch.poisson(x_end/extract(self.lmd, t, x_start.shape)))
        # xt = (extract(self.alphas_cumprod, t, x_start.shape) * torch.poisson(x_start*extract(self.lmd, t, x_start.shape))/extract(self.lmd, t, x_start.shape) +
        #       extract(self.one_minus_alphas_cumprod, t, x_start.shape) *x_end)
        xt = (extract(self.alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.one_minus_alphas_cumprod, t, x_start.shape) * x_end)

        return torch.clip(xt,0,1)


    def get_x2_bar_from_xt(self, x1_bar, xt, t):

        a = (xt - extract(self.alphas_cumprod, t, x1_bar.shape) * x1_bar)/extract(self.one_minus_alphas_cumprod, t, x1_bar.shape)

        return (a-a.min())/(a.max()-a.min())


    @torch.no_grad()
    def sample(self, batch_size=1, img=None, t=None,low_dose = None,sampling_routine='ddim', n_iter=1, start_adjust_iter=1):
        self.denoise_fn.eval()
        self.admmnet.eval()
        if t == None:
            t = self.num_timesteps
        if self.context:
            up_img = img[:, 0].unsqueeze(1)
            down_img = img[:, 2].unsqueeze(1)
            img = img[:, 1].unsqueeze(1)

        noise = img #img 是lowdose_proj
        low_dose = low_dose #是lowdose

        x1_bar = img
        direct_recons = []
        diff_imgs = []
        admm_imgs= []
        t_fortseting = []

        if sampling_routine == 'ddim':
            while (t):
                t_fortseting.append(t)

                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)

                if self.context:
                    full_img = torch.cat((up_img, img, down_img), dim=1)
                else:
                    full_img = img

                if t == self.num_timesteps or n_iter < start_adjust_iter:
                    adjust = False
                else:
                    adjust = True

                x1_bar = self.denoise_fn(full_img, step, x1_bar, noise, adjust=adjust) # 后两个输入 x1_bar 和 noise 其实没用上
                # x1_bar = torch.clamp(x1_bar, 0)
                print(x1_bar.min().item(),x1_bar.max().item(),"\t")
                direct_recons.append(x1_bar)
                # x1_bar,_,_ = self.admmnet(x1_bar,noise)
                # x1_bar = torch.clamp(x1_bar,0)
                print(x1_bar.min().item(),x1_bar.max().item(),"\t")
                admm_imgs.append(x1_bar)


                x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step) #取t时刻的扩散终点

                xt_bar = x1_bar
                if t != 0:
                    # if xt_bar.min().item() < 0 or x2_bar.min().item()<0:
                    #     print("ddim\t\t",xt_bar.min().item(), x2_bar.min().item())
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    # if xt_sub1_bar.min().item() < 0 or x2_bar.min().item() < 0 :
                    #     print("ddim_t_not1\t\t", xt_sub1_bar.min().item(), x2_bar.min().item())
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                img = img - xt_bar + xt_sub1_bar
                # img1,_,_ = self.admmnet(img,img)
                img = torch.clamp(img,0)

                diff_imgs.append(img)


                t = t - 1

        elif sampling_routine == 'x0_step_down':
            while (t):
                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)

                if self.context:
                    full_img = torch.cat((up_img, img, down_img), dim=1)
                else:
                    full_img = img

                if t == self.num_timesteps:
                    adjust = False
                else:
                    adjust = True

                x1_bar = self.denoise_fn(full_img, step, x1_bar, noise, adjust=adjust)
                x2_bar = noise

                xt_bar = x1_bar
                if t != 0:
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                img = img - xt_bar + xt_sub1_bar

                direct_recons.append(x1_bar)
                admm_imgs.append(x2_bar)
                diff_imgs.append(img)
                t = t - 1
        elif sampling_routine == 'cold':
            x_T = img
            adjust = True
            x2_bar = []
            for i in range(t):
                if t-i-1 >0:
                    step = torch.full((batch_size,), t-i, dtype=torch.long, device=img.device)
                    xt_bar = self.denoise_fn(x_T, step, x1_bar, noise, adjust=adjust) - x_T

                    step_1 = torch.full((batch_size,), t-i-1, dtype=torch.long, device=img.device)
                    xt_sub1_bar = self.denoise_fn(x_T, step_1, x1_bar, noise, adjust=adjust) - x_T
                    xk_1 = x_T - xt_bar + xt_sub1_bar
                    x_T = xk_1
                    x1_bar = xk_1

                    direct_recons.append(xk_1)
                    diff_imgs.append(xt_sub1_bar)
                else:
                    return xk_1.clamp(0., 1.), torch.stack(direct_recons), torch.stack(diff_imgs)

        if sampling_routine == 'sino':
            diff_imgs = []
            direct_recons = []
            admm_imgs = []
            while (t):
                t_fortseting.append(t)

                step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)

                if self.context:
                    full_img = torch.cat((up_img, img, down_img), dim=1)
                else:
                    full_img = img

                if t == self.num_timesteps or n_iter < start_adjust_iter:
                    adjust = False
                else:
                    adjust = True

                x1_bar = self.denoise_fn(full_img, step, x1_bar, noise, adjust=adjust) # 后两个输入 x1_bar 和 noise 其实没用上
                # x1_bar = torch.clamp(x1_bar, 0)
                # print(x1_bar.min().item(),x1_bar.max().item(),"\t")

                admm_recon,_,_ = self.admmnet(x1_bar,noise,low_dose,step)
                direct_recons.append(admm_recon)
                # x1_bar = torch.clamp(x1_bar,0)
                # print(x1_bar.min().item(),x1_bar.max().item(),"\t")



                x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step) #取t时刻的扩散终点

                xt_bar = x1_bar
                if t != 0:
                    # if xt_bar.min().item() < 0 or x2_bar.min().item()<0:
                    #     print("ddim\t\t",xt_bar.min().item(), x2_bar.min().item())
                    xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

                xt_sub1_bar = x1_bar
                if t - 1 != 0:
                    step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                    # if xt_sub1_bar.min().item() < 0 or x2_bar.min().item() < 0 :
                    #     print("ddim_t_not1\t\t", xt_sub1_bar.min().item(), x2_bar.min().item())
                    xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

                img = img - xt_bar + xt_sub1_bar
                diff_imgs.append(img)
                img1,_,_ = self.admmnet(img,noise,low_dose,step)
                admm_imgs.append(img1)
                # img = torch.clamp(img,0)
                t = t - 1
        # return img.clamp(0., 1.), torch.stack(direct_recons), torch.stack(diff_imgs),torch.stack(admm_imgs)
        return img1, torch.stack(direct_recons), torch.stack(diff_imgs),torch.stack(admm_imgs),admm_recon

    def forward(self, x, y, n_iter,low_dose, only_adjust_two_step=True, start_adjust_iter=1):
        '''
        :param x: low dose image
        :param y: ground truth image
        :param n_iter: trainging iteration
        :param only_adjust_two_step: only use the EMM module in the second stage. Default: True
        :param start_adjust_iter: the number of iterations to start training the EMM module. Default: 1
        '''
        if only_adjust_two_step:
            print("only_adjust_two_step is true")

        # b, c, h, w, device, img_size, = *y.shape, y.device, self.image_size
        b, c, h, w = y.shape
        device = y.device
        img_size = self.image_size

        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        
        t_single = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        t = t_single.repeat((b,))

# 只有cold diffusion
        x_end = x[:, 0].unsqueeze(1)
        x_mix = self.q_sample(x_start=y, x_end=x_end, t=t)  # qsample 均值不变加噪函数，需要起点，终点，和时间步t，三个输入
        x_recon = self.denoise_fn(x_mix, t, y, x_end, adjust=False)  # x_mix 即为扩散加噪的图，y为gt，x_end为input
        # print("stage I", x_recon.min().item(), x_recon.max().item())

        x_recon, diff_recon, sinodiff = self.admmnet(x_recon,x_end,low_dose,t)
        # print("stage I after admm", x_recon.min().item(), x_recon.max().item())
        if t_single.item() >= 1:
            t_sub1 = t - 1
            t_sub1[t_sub1 < 0] = 0
            x_mix_sub1 = self.q_sample(x_start=diff_recon, x_end=x_end, t=t_sub1)
            x_recon_sub1 = self.denoise_fn(x_mix_sub1, t_sub1, diff_recon, x_end, adjust=True)
            # print("stage II", x_recon_sub1.min().item(), x_recon_sub1.max().item())
            x_recon_sub1, diff_recon_sub1,sinodiff_sub1 = self.admmnet(x_recon_sub1, x_end,low_dose,t_sub1)
            # print("stage II after admm", x_recon_sub1.min().item(), x_recon_sub1.max().item())
        else:
            x_recon_sub1, x_mix_sub1, diff_recon_sub1,sinodiff_sub1 = x_recon, x_mix, diff_recon,sinodiff


# ###作者的方法
#         if self.context:
#             x_end = x[:,1].unsqueeze(1)
#             x_mix = self.q_sample(x_start=y, x_end=x_end, t=t) # qsample 均值不变加噪函数，需要起点，终点，和时间步t，三个输入
#             x_mix = torch.cat((x[:,0].unsqueeze(1), x_mix, x[:,2].unsqueeze(1)), dim=1)
#         else:
#             x_end = x
#             if y.min().item()<0:
#                 print("fowward\t\t", y.min().item(), x_end.min().item())
#             x_mix = self.q_sample(x_start=y, x_end=x_end, t=t)
#
#         # stage I
#         if only_adjust_two_step or n_iter < start_adjust_iter:
#             x_recon = self.denoise_fn(x_mix, t, y, x_end, adjust=False) # x_mix 即为扩散加噪的图，y为gt，x_end为input
#             # x_recon,diff_recon,sinodiff = self.admmnet(x_recon)
#
#         else:
#             if t[0] == self.num_timesteps - 1:
#                 adjust = False
#             else:
#                 adjust = True
#             x_recon = self.denoise_fn(x_mix, t, y, x_end, adjust=adjust)
#             # print("stage I ",x_recon.min().item(), x_recon.max().item())
#             # x_recon,diff_recon,sinodiff = self.admmnet(x_recon)
#
#
#         # stage II
#         if n_iter >= start_adjust_iter and t_single.item() >= 1:
#             t_sub1 = t - 1
#             t_sub1[t_sub1 < 0] = 0
#
#             if self.context:
#                 x_mix_sub1 = self.q_sample(x_start=x_recon, x_end=x_end, t=t_sub1)
#                 x_mix_sub1 = torch.cat((x[:, 0].unsqueeze(1), x_mix_sub1, x[:, 2].unsqueeze(1)), dim=1)
#             else:
#                 # if x_recon.min().item() < 0:
#                     # print("stageII\t\t", x_recon.min().item(), x_end.min().item()) 注意此处小于0 了
#                 x_mix_sub1 = self.q_sample(x_start=x_recon, x_end=x_end, t=t_sub1)
#
#             x_recon_sub1 = self.denoise_fn(x_mix_sub1, t_sub1, x_recon, x_end, adjust=True)
#
#
#         else:
#             x_recon_sub1, x_mix_sub1 = x_recon, x_mix

        return x_recon,x_mix,diff_recon, x_recon_sub1, x_mix_sub1, diff_recon_sub1,sinodiff,sinodiff_sub1
