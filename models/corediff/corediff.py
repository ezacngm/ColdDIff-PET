import os.path as osp
from torch.nn import functional as F
import torch
import torchvision
import os
import argparse
import tqdm
import itertools
import copy
from utils.measure import *
from utils.loss_function import PerceptualLoss
from utils.ema import EMA
from torchvision.utils import save_image
from models.basic_template import TrainTask
from .corediff_wrapper import Network, WeightNet
from .diffusion_modules import Diffusion
import datetime
import wandb
from pytorch_msssim import SSIM
import time
from .admm_v3 import admm
from torch_radon import ParallelBeam


output_path = r'C:\Users\Administrator\Desktop\CoreDiff-main\output'
diffusionprocess_images_dir = r"C:\Users\Administrator\Desktop\CoreDiff-main\output\diffusion_process"
pbeam = ParallelBeam(128, np.linspace(0, np.pi, 128, endpoint=False))


def normalize_(img, min_val=None, max_val=None):
    min_val = img.min()
    max_val = img.max()
    # Normalize PET images to a [0, 1] range
    img_normalized = (img - min_val) / (max_val - min_val)
    # Clip values just in case to ensure they remain within [0, 1]
    img_normalized = img_normalized.clip(0, 1)
    # img = transforms.RandomRotation(degrees=(-15, 15))(img)  # todo: 看torch文档 要求输入shape是 BCWH
    # img = transforms.RandomHorizontalFlip(p=0.5)(img)
    return img_normalized


class corediff(TrainTask):
    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument("--in_channels", default=1, type=int)
        parser.add_argument("--out_channels", default=1, type=int)
        parser.add_argument("--init_lr", default=5e-5, type=float)

        parser.add_argument('--update_ema_iter', default=10, type=int)
        parser.add_argument('--start_ema_iter', default=2000, type=int)
        parser.add_argument('--ema_decay', default=0.995, type=float)

        parser.add_argument('--T', default=10, type=int)

        parser.add_argument('--sampling_routine', default='ddim', type=str)
        parser.add_argument('--only_adjust_two_step', action='store_true')
        parser.add_argument('--start_adjust_iter', default=1, type=int)
        
        return parser

    def set_model(self):
        opt = self.opt
        self.ema = EMA(opt.ema_decay)
        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.dose = opt.dose
        self.T = opt.T
        self.sampling_routine = opt.sampling_routine
        self.context = opt.context
        
        denoise_fn = Network(in_channels=opt.in_channels, context=opt.context)
        admmnet = admm()
        model = Diffusion(
            denoise_fn=denoise_fn,
            admmnet=admmnet,
            image_size=128, ## 注意这个地方即使加了--image_size=256 也要改
            timesteps=opt.T,
            context=opt.context
        ).cuda()

        parameters = itertools.chain(model.denoise_fn.parameters(),model.admmnet.parameters())
        # optimizer = torch.optim.Adam(parameters, opt.init_lr)

        optimizer = torch.optim.Adam(model.denoise_fn.parameters(), opt.init_lr)
        optimizer_admm = torch.optim.Adam(model.admmnet.parameters(), opt.init_lr/2)
        # ema_model = copy.deepcopy(model)
        self.ssimmodule = SSIM(data_range=1, size_average=True, channel=1)
        # self.logger.modules = [model, ema_model, optimizer] #在此处保存模型 admm也保存上
        self.logger.modules = [model.denoise_fn, model.admmnet, optimizer, optimizer_admm] #在此处传要保存的模型名字 admm也保存上

        self.model = model
        self.optimizer = optimizer
        self.optimizer_admm =optimizer_admm
        self.ema_model = model

        self.lossfn = nn.MSELoss()
        self.l1loss = nn.SmoothL1Loss()
        self.huber = nn.HuberLoss(delta=0.001)
        self.lossfn_sub1 = nn.MSELoss()

        self.reset_parameters()

    def clip_grad_norm(self,optimizer, max_norm, norm_type=2):
        """
        Clip the norm of the gradients for all parameters under `optimizer`.
        Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
        """
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())


    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def addnoise(self,pbeam, img, count):
        proj = pbeam.forward(img)  # 投影到sinogram
        mul_factor = torch.ones_like(proj)
        mul_factor = mul_factor + (torch.rand_like(mul_factor) * 0.2 - 0.1)
        noise = torch.ones_like(proj) * torch.mean(mul_factor * proj, dim=(-1, -2), keepdims=True) * 0.2
        sino = mul_factor * proj + noise
        cs = count / (1e-9 + torch.sum(sino, dim=(-1, -2), keepdim=True))
        sino = sino * cs
        mul_factor = mul_factor * cs
        noise = noise * cs
        x = torch.poisson(sino)
        sinogram = nn.ReLU()((x - noise) / mul_factor)
        #     sinogram = pbeam.filter_sinogram(sinogram)
        img = pbeam.backward(sinogram)
        return img, sinogram

    def train(self, inputs, n_iter):
        opt = self.opt
        self.model.train()
        self.ema_model.train()
        low_dose, full_dose = inputs
        low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
        low_dose_proj = pbeam.forward(low_dose)
        low_dose_proj = normalize_(low_dose_proj)
        full_dose_proj = pbeam.forward(full_dose)
        full_dose_proj = normalize_(full_dose_proj)

        psnr, ssim, rmse = 0., 0., 0.
        psnr_list =[]
        ssim_list = []
        ## training process of CoreDiff
        gen_full_dose, x_mix, diff_recon,gen_full_dose_sub1, x_mix_sub1,diff_recon_sub1,sinodiff,sinodiff_sub1 = self.model(
            low_dose_proj, full_dose_proj, n_iter,low_dose,
            only_adjust_two_step=opt.only_adjust_two_step,
            start_adjust_iter=opt.start_adjust_iter
        )
        data_range = full_dose.max()
        data_range_sino = full_dose_proj.max()
        self.ssimmodule = SSIM(data_range=data_range, size_average=True, channel=1)
        self.ssimmodule_sino = SSIM(data_range=data_range_sino, size_average=True, channel=1)
        # print("genfulldose value scope",gen_full_dose.min().item(), gen_full_dose.max().item())


        #------  训练diffusion
        proj = pbeam.forward(full_dose)
        proj_low = pbeam.forward(low_dose)
        loss_l2_admm = self.lossfn(gen_full_dose, full_dose) + self.lossfn(gen_full_dose_sub1,full_dose)
        loss_huber_admm = self.huber(gen_full_dose, full_dose) + self.huber(gen_full_dose_sub1,full_dose)
        # loss = (0.5 * self.lossfn(gen_full_dose, full_dose) + 0.5 * self.lossfn_sub1(gen_full_dose_sub1, full_dose))
        loss_ssim_admm =1-self.ssimmodule(full_dose, gen_full_dose) + 1-self.ssimmodule(full_dose, gen_full_dose_sub1)
        loss_sino_admm = self.lossfn(sinodiff, torch.zeros_like(sinodiff)) + self.lossfn(sinodiff_sub1,torch.zeros_like(sinodiff_sub1))
        loss_ssim_diff = (1 - self.ssimmodule_sino(diff_recon, full_dose_proj)) + (
                    1 - self.ssimmodule_sino(diff_recon_sub1, full_dose_proj))
        loss_l2_diff = self.lossfn(diff_recon,full_dose_proj) + self.lossfn(diff_recon_sub1,full_dose_proj)
        loss_l1_diff = self.l1loss(diff_recon, full_dose_proj) + self.l1loss(diff_recon_sub1, full_dose_proj)
        loss_admm = loss_l2_admm +loss_sino_admm
        loss_diff = loss_l2_diff+0.5*loss_l1_diff+loss_ssim_diff
        # loss = loss_l2_admm  + loss_l2_diff + loss_ssim_admm + loss_ssim_diff

        loss =  loss_diff

        self.model.denoise_fn.train(True)
        self.model.admmnet.train(False)
        self.optimizer.zero_grad()
        self.optimizer_admm.zero_grad()
        loss.backward()
        # self.clip_grad_norm(self.optimizer_admm, 1.0)
        # self.clip_grad_norm(self.optimizer, 1.0)
        self.optimizer.step()



        # ------  训练admm
        gen_full_dose, x_mix, diff_recon,gen_full_dose_sub1, x_mix_sub1,diff_recon_sub1,sinodiff,sinodiff_sub1 = self.model(
            low_dose_proj, full_dose_proj, n_iter,low_dose,
            only_adjust_two_step=opt.only_adjust_two_step,
            start_adjust_iter=opt.start_adjust_iter
        )
        proj_gt = pbeam.forward(full_dose)
        proj_low = pbeam.forward(low_dose)
        loss_l2_admm = self.lossfn(gen_full_dose, full_dose) + self.lossfn(gen_full_dose_sub1, full_dose)
        loss_l1_admm = self.l1loss(gen_full_dose, full_dose) + self.l1loss(gen_full_dose_sub1, full_dose)
        # loss = (0.5 * self.lossfn(gen_full_dose, full_dose) + 0.5 * self.lossfn_sub1(gen_full_dose_sub1, full_dose))
        loss_ssim_admm = 1 - self.ssimmodule(full_dose, gen_full_dose) + 1 - self.ssimmodule(full_dose,
                                                                                             gen_full_dose_sub1)
        loss_sino_admm = self.lossfn(sinodiff, proj_gt) + self.lossfn(sinodiff_sub1, proj_gt)
        loss_ssim_diff = (1 - self.ssimmodule(diff_recon, full_dose)) + (
                1 - self.ssimmodule(diff_recon_sub1, full_dose))
        loss_l2_diff = self.lossfn(diff_recon, full_dose_proj) + self.lossfn(diff_recon_sub1, full_dose_proj)
        loss_admm = loss_l2_admm + 0.5*loss_ssim_admm
        loss_diff = loss_l2_diff
        # loss = loss_l2_admm  + loss_l2_diff + loss_ssim_admm + loss_ssim_diff

        loss = loss_admm

        self.model.denoise_fn.train(False)
        self.model.admmnet.train(True)
        self.optimizer.zero_grad()
        self.optimizer_admm.zero_grad()
        loss.backward()
        self.clip_grad_norm(self.optimizer_admm, 1.0)
        self.optimizer_admm.step()

        if opt.wandb:
            if n_iter == opt.resume_iter + 1:
                wandb.init(project="your wandb project name")

        lr = self.optimizer.param_groups[0]['lr']
        loss_diff = loss_l2_diff.item()
        loss_admm = loss_l2_admm.item()
        # self.logger.msg([loss, lr], n_iter)
        # tqdm.tqdm.write(f" {n_iter:05d}, loss {loss:.5f}, lr {lr:.5f}")



        if n_iter % self.update_ema_iter == 0:
            self.step_ema(n_iter)



        # full_dose = self.transfer_calculate_window(full_dose)
        # gen_full_dose = self.transfer_calculate_window(gen_full_dose)
        data_range = full_dose.max() - full_dose.min()

        psnr_score_admm, ssim_score_admm, rmse_score_admm = compute_measure(full_dose, gen_full_dose, data_range)
        psnr_score_diff, ssim_score_diff, rmse_score_diff = compute_measure(full_dose_proj, diff_recon, data_range)
        if n_iter%200==0 and n_iter != 0:
            bb = torch.cat((low_dose_proj, full_dose_proj, x_mix, gen_full_dose,diff_recon,low_dose,full_dose),0)
            aa = torchvision.utils.make_grid(bb,nrow=5,normalize=True)
            torchvision.utils.save_image(aa, r"C:\Users\Administrator\Desktop\genfulldose.png")

        if opt.wandb:
            wandb.log({'epoch': n_iter, 'loss': loss})
        return loss_diff, lr, gen_full_dose, low_dose, ssim_score_diff, ssim_score_admm,loss_admm

    @torch.no_grad()
    def test(self, n_iter):
        opt = self.opt
        self.ema_model.eval()
        ssim_combined = []
        ssim_onlyadmm = []
        ssim_onlydiff = []
        i = 0
        i_max = 30
        psnr, ssim, rmse = 0., 0., 0.
        psnr_admm, ssim_admm, rmse_admm = 0., 0., 0.
        progress_bar = tqdm.tqdm(self.test_loader, desc='test', leave=True, position=0)
        for low_dose, full_dose in progress_bar:
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
            low_dose_proj = pbeam.forward(low_dose)
            low_dose_proj = normalize_(low_dose_proj)
            full_dose_proj = pbeam.forward(full_dose)
            full_dose_proj = normalize_(full_dose_proj)


            gen_full_dose, direct_recons, diff_imgs,admm_imgs,gen_admm = self.ema_model.sample(
                batch_size = low_dose.shape[0],
                img = low_dose_proj,
                t = self.T,
                low_dose=low_dose,
                sampling_routine = self.sampling_routine,
                n_iter=n_iter,
                start_adjust_iter=opt.start_adjust_iter,
            )
            # print(full_dose.min(), full_dose.max())
            full_dose = self.transfer_calculate_window(full_dose)
            # print(full_dose.min(), full_dose.max())
            # print(gen_full_dose.min(), gen_full_dose.max())
            gen_full_dose = self.transfer_calculate_window(gen_full_dose)
            # print(gen_full_dose.min(), gen_full_dose.max())
            # print(gen_admm.min(), gen_admm.max())
            gen_admm = self.transfer_calculate_window(gen_admm)
            # print(gen_admm.min(), gen_admm.max())

            data_range = max(full_dose.max() - full_dose.min(),gen_full_dose.max() - gen_full_dose.min())
            data_range_admm = max(full_dose.max() - full_dose.min(), gen_admm.max() - gen_admm.min())
            psnr_score, ssim_score, rmse_score = compute_measure(full_dose, gen_full_dose, data_range)
            psnr_score_admm, ssim_score_admm, rmse_score_admm = compute_measure(full_dose, gen_admm, data_range_admm)
            psnr += psnr_score / len(self.test_loader)
            ssim += ssim_score / len(self.test_loader)
            rmse += rmse_score / len(self.test_loader)
            psnr_admm += psnr_score_admm / len(self.test_loader)
            ssim_admm += ssim_score_admm / len(self.test_loader)
            rmse_admm += rmse_score_admm / len(self.test_loader)

            ssim_combined.append(ssim)
            ssim_onlyadmm.append(ssim_admm)


            progress_bar.set_postfix(

                psnr=psnr_score,
                ssim = ssim_score,
                acc_psnr = psnr,
                acc_ssim = ssim,
                acc_psnr_admm = psnr_admm,
                acc_ssim_admm = ssim_admm

            )
            # psnr += psnr_score / i_max
            # ssim += ssim_score / i_max
            # rmse += rmse_score / i_max

            # i+=1
            # if i > i_max:
            #     self.logger.msg([psnr, ssim, rmse], n_iter)
            #     break



        if opt.wandb:
            wandb.log({'epoch': n_iter, 'PSNR': psnr, 'SSIM': ssim, 'RMSE': rmse})
        self.logger.msg([psnr, ssim, rmse], n_iter)
        self.logger.msg([psnr_admm, ssim_admm, rmse_admm], n_iter)
        np.save(r"C:\Users\Administrator\Desktop\ssim_onlydiff.npy",ssim_combined)
        np.save(r"C:\Users\Administrator\Desktop\ssim_onlydiff.npy", ssim_onlyadmm)

    @torch.no_grad()
    def generate_images(self, n_iter):
        opt = self.opt
        self.ema_model.eval()
        low_dose, full_dose = self.test_images
        low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
        low_dose_proj = pbeam.forward(low_dose)
        low_dose_proj = normalize_(low_dose_proj)
        full_dose_proj = pbeam.forward(full_dose)
        full_dose_proj = normalize_(full_dose_proj)
        gen_full_dose, direct_recons, diff_imgs, admm_imgs,gen_admm = self.ema_model.sample(
                batch_size = low_dose.shape[0],
                img = low_dose_proj,
                t = self.T,
                low_dose = low_dose,
                sampling_routine = self.sampling_routine,
                n_iter=n_iter,
                start_adjust_iter=opt.start_adjust_iter,
            )

        if self.context:
            low_dose = low_dose[:, 1].unsqueeze(1)

        b, c, w, h = low_dose.size()
        fake_imgs = torch.stack([low_dose, full_dose, gen_full_dose, gen_admm])
        # fake_imgs = self.transfer_display_window(fake_imgs)
        fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, c, w, h))
        save_image(torchvision.utils.make_grid(diff_imgs[:,0,:,:,:].squeeze(2), nrow=8,normalize=True),
                   r"C:\Users\Administrator\Desktop\diff_imgs.png")  ### 这一步存了个照片 训练的时候

        save_image(torchvision.utils.make_grid(direct_recons[:,0,:,:,:].squeeze(2), nrow=8,normalize=True),
                   r"C:\Users\Administrator\Desktop\direct_recon.png")  ### 这一步存了个照片 训练的时候
        save_image(torchvision.utils.make_grid(admm_imgs[:,0,:,:,:].squeeze(2), nrow=8,normalize=True),
                   r"C:\Users\Administrator\Desktop\admm_imgs.png")  ### 这一步存了个照片 训练的时候
        self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=3,normalize=True),
                               n_iter, '4.15test_{}_{}'.format(self.dose, self.sampling_routine) + '_' + opt.test_dataset)
    @torch.no_grad()
    # def generate_images(self, n_iter):
    #     opt = self.opt
    #     self.ema_model.eval()
    #     for i,images in enumerate(self.test_loader):
    #         low_dose, full_dose = images
    #         low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
    #         low_dose_proj = pbeam.forward(low_dose)
    #         low_dose_proj = normalize_(low_dose_proj)
    #         full_dose_proj = pbeam.forward(full_dose)
    #         full_dose_proj = normalize_(full_dose_proj)
    #         gen_full_dose, direct_recons, diff_imgs, admm_imgs,gen_admm = self.ema_model.sample(
    #                 batch_size = low_dose.shape[0],
    #                 img = low_dose_proj,
    #                 t = self.T,
    #                 low_dose = low_dose,
    #                 sampling_routine = self.sampling_routine,
    #                 n_iter=n_iter,
    #                 start_adjust_iter=opt.start_adjust_iter,
    #             )
    #
    #         if self.context:
    #             low_dose = low_dose[:, 1].unsqueeze(1)
    #
    #         b, c, w, h = low_dose.size()
    #         fake_imgs = torch.stack([low_dose, full_dose, gen_full_dose, gen_admm])
    #         # fake_imgs = self.transfer_display_window(fake_imgs)
    #         fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, c, w, h))
    #         # save_image(torchvision.utils.make_grid(diff_imgs[:,0,:,:,:].squeeze(2), nrow=8,normalize=True),
    #         #            r"C:\Users\Administrator\Desktop\diff_imgs.png")  ### 这一步存了个照片 训练的时候
    #         #
    #         # save_image(torchvision.utils.make_grid(direct_recons[:,0,:,:,:].squeeze(2), nrow=8,normalize=True),
    #         #            r"C:\Users\Administrator\Desktop\direct_recon.png")  ### 这一步存了个照片 训练的时候
    #         # save_image(torchvision.utils.make_grid(admm_imgs[:,0,:,:,:].squeeze(2), nrow=8,normalize=True),
    #         #            r"C:\Users\Administrator\Desktop\admm_imgs.png")  ### 这一步存了个照片 训练的时候
    #         # self.logger.save_image(torchvision.utils.make_grid(gen_admm, nrow=3,normalize=True),
    #         #                        n_iter, 'test_{}_{}_slice{:03d}'.format(self.dose, self.sampling_routine, i ) + '_' + opt.test_dataset)
    #
    #         save_image(fake_imgs, os.path.join(r"C:\Users\Administrator\Desktop\output","test_{}_{}_slice{:03d}.png".format(self.dose, self.sampling_routine, i )))

    def train_osl_framework(self, test_iter):
        opt = self.opt
        self.ema_model.eval()

        ''' Initialize WeightNet '''
        weightnet = WeightNet(weight_num=20).cuda()
        optimizer_w = torch.optim.Adam(weightnet.parameters(), opt.init_lr*10)
        lossfn = PerceptualLoss()

        ''' get imstep images of diffusion '''
        for i in range(len(self.test_dataset)-2):
            if i == opt.index:
                if opt.unpair:
                    low_dose, _ = self.test_dataset[i]
                    _, full_dose = self.test_dataset[i+2]
                else:
                    low_dose, full_dose = self.test_dataset[i]
        low_dose, full_dose = torch.from_numpy(low_dose).unsqueeze(0).cuda(), torch.from_numpy(full_dose).unsqueeze(0).cuda()

        gen_full_dose, direct_recons, diff_imgs,admm_imgs,gen_admm = self.ema_model.sample(
            batch_size=low_dose.shape[0],
            img=low_dose,
            t=self.T,
            sampling_routine=self.sampling_routine,
            start_adjust_iter=opt.start_adjust_iter,
        )

        inputs = diff_imgs.transpose(0, 2).squeeze(0)
        targets = full_dose

        ''' train WeightNet '''
        input_patches, target_patches = self.get_patch(inputs, targets, patch_size=opt.patch_size, stride=32)
        input_patches, target_patches = input_patches.detach(), target_patches.detach()
        progress_bar = tqdm.trange(1, opt.osl_max_iter)
        for n_iter in progress_bar:
            weightnet.train()
            batch_ids = torch.from_numpy(np.random.randint(0, input_patches.shape[0], opt.osl_batch_size)).cuda()
            input = input_patches.index_select(dim = 0, index = batch_ids).detach()
            target = target_patches.index_select(dim = 0, index = batch_ids).detach()

            out, weights = weightnet(input)
            # print(out.min(),out.max())
            # print(target.min(), target.max())
            # loss = lossfn(out, target) + 0.3*(1-self.ssimmodule(out, target))
            # loss = lossfn(out, target)
            loss =   0.3*(1 - self.ssimmodule(out, target))+lossfn(out, target)
            loss.backward()

            optimizer_w.step()
            optimizer_w.zero_grad()
            lr = optimizer_w.param_groups[0]['lr']
            # self.logger.msg([loss, lr], n_iter)
            progress_bar.set_postfix(

                loss=loss.item(),
                lr=lr,

            )
            if opt.wandb:
                wandb.log({'epoch': n_iter, 'loss': loss})
        opt_image = weights * inputs
        opt_image = opt_image.sum(dim=1, keepdim=True)
        print(weights)

        ''' Calculate the quantitative metrics before and after weighting'''
        full_dose_cal = self.transfer_calculate_window(full_dose)
        gen_full_dose_cal = self.transfer_calculate_window(gen_full_dose)
        opt_image_cal = self.transfer_calculate_window(opt_image)
        data_range = full_dose_cal.max() - full_dose_cal.min()
        psnr_ori, ssim_ori, rmse_ori = compute_measure(full_dose_cal, gen_full_dose_cal, data_range)
        psnr_opt, ssim_opt, rmse_opt = compute_measure(full_dose_cal, opt_image_cal, data_range)
        self.logger.msg([psnr_ori, ssim_ori, rmse_ori], test_iter)
        self.logger.msg([psnr_opt, ssim_opt, rmse_opt], test_iter)
        if self.context:
            print("context is true")
            fake_imgs = torch.cat((low_dose[:, 1].unsqueeze(1), full_dose, gen_full_dose, opt_image), dim=0)
        else:
            # Assuming full_dose, gen_full_dose, and opt_image are compatible for concatenation with low_dose
            fake_imgs = torch.cat((low_dose, full_dose, gen_full_dose, opt_image), dim=0)

        fake_imgs = self.transfer_display_window(fake_imgs)
        self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=4), test_iter,
                               'test_opt_' + opt.test_dataset + '_dose{}_slice{}'.format(self.dose, opt.index))### 这一步存了个照片 训练的时候

        if opt.unpair:
            weights_dir = os.path.join(output_path, "unpaired_weights")
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)
            filename = os.path.join(output_path,"unpaired_weights",opt.test_dataset+'_{}_{}_).npy'.format(self.dose, opt.index))
        else:
            weights_dir = os.path.join(output_path, "paired_weights")
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)
            filename = os.path.join(output_path,"paired_weights",opt.test_dataset+'_{}_{}.npy'.format(self.dose, opt.index))
        np.save(filename, weights.detach().cpu().squeeze().numpy())


    def test_osl_framework(self, test_iter):
        opt = self.opt
        self.ema_model.eval()

        if opt.unpair:
            weights_dir = os.path.join(output_path, "unpaired_weights")
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)
            filename = os.path.join(output_path, "unpaired_weights",
                                    opt.test_dataset + '_{}_{}.npy'.format(self.dose, opt.index))
        else:
            weights_dir = os.path.join(output_path, "paired_weights")
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)
            filename = os.path.join(output_path, "paired_weights",
                                    opt.test_dataset + '_{}_{}.npy'.format(self.dose, opt.index))
        weights = np.load(filename)
        print(weights)
        weights = torch.from_numpy(weights).unsqueeze(1).unsqueeze(2).unsqueeze(0).cuda()

        psnr_ori, ssim_ori, rmse_ori = 0., 0., 0.
        psnr_opt, ssim_opt, rmse_opt = 0., 0., 0.


        progress_bar = tqdm.tqdm(self.test_loader, desc='test')
        i_log=0
        for low_dose, full_dose in progress_bar:
            i_log+=1
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

            gen_full_dose, direct_recons, diff_imgs,admm_imgs = self.ema_model.sample(
                batch_size=low_dose.shape[0],
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                n_iter=test_iter,
                start_adjust_iter=opt.start_adjust_iter,
            )
            # print(diff_imgs.shape)
            #存扩散过程图像
            if i_log%20==0:
                save_image(torchvision.utils.make_grid(diff_imgs.squeeze(2), nrow=4),
                           os.path.join(diffusionprocess_images_dir,'diff_proc_' + opt.test_dataset + '_dose{}_slice{}_{}.png'.format(self.dose, opt.index, i_log)))  ### 这一步存了个照片 训练的时候
            diff_imgs = diff_imgs[:self.T]
            inputs = diff_imgs.squeeze(2).transpose(0, 1)
            # print(diff_imgs.shape)
            opt_image = weights * inputs

            # 存加权图像：
            if i_log%20==0:
                save_image(torchvision.utils.make_grid(opt_image.transpose(0, 1), nrow=4),
                           os.path.join(diffusionprocess_images_dir,'weighted_average' + opt.test_dataset + '_dose{}_slice{}_{}.png'.format(self.dose, opt.index, i_log)))  ### 这一步存了个照片 训练的时候
            opt_image = opt_image.sum(dim=1, keepdim=True)

            full_dose = self.transfer_calculate_window(full_dose)
            gen_full_dose = self.transfer_calculate_window(gen_full_dose)
            opt_image = self.transfer_calculate_window(opt_image)

            data_range = full_dose.max() - full_dose.min()
            psnr_ori, ssim_ori, rmse_ori = compute_measure(full_dose, gen_full_dose, data_range)
            psnr_opt, ssim_opt, rmse_opt = compute_measure(full_dose, opt_image, data_range)

            psnr_ori += psnr_ori / len(self.test_loader)
            ssim_ori += ssim_ori / len(self.test_loader)
            rmse_ori += rmse_ori / len(self.test_loader)

            psnr_opt += psnr_opt / len(self.test_loader)
            ssim_opt += ssim_opt / len(self.test_loader)
            rmse_opt += rmse_opt / len(self.test_loader)

        self.logger.msg([psnr_ori, ssim_ori, rmse_ori], test_iter)
        self.logger.msg([psnr_opt, ssim_opt, rmse_opt], test_iter)

        progress_bar.set_postfix(

            psnr_ori=psnr_ori,
            ssim_ori=ssim_ori,
            rmse_ori=rmse_ori,
            psnr_opt=psnr_opt,
            ssim_opt=ssim_opt,
            rmse_opt=rmse_opt,

        )

    def get_patch(self, input_img, target_img, patch_size=220, stride=4):
        input_patches = []
        target_patches = []
        _, c_input, h, w = input_img.shape
        _, c_target, h, w = target_img.shape

        Top = np.arange(0, h - patch_size + 1, stride)
        Left = np.arange(0, w - patch_size + 1, stride)
        for t_idx in range(len(Top)):
            top = Top[t_idx]
            for l_idx in range(len(Left)):
                left = Left[l_idx]
                input_patch = input_img[:, :, top:top + patch_size, left:left + patch_size]
                target_patch = target_img[:, :, top:top + patch_size, left:left + patch_size]
                input_patches.append(input_patch)
                target_patches.append(target_patch)

        input_patches = torch.stack(input_patches).transpose(0, 1).reshape((-1, c_input, patch_size, patch_size))
        target_patches = torch.stack(target_patches).transpose(0, 1).reshape((-1, c_target, patch_size, patch_size))
        return input_patches, target_patches
