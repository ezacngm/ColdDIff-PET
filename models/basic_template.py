# This part builds heavily on https://github.com/Hzzone/DU-GAN.
import torch
import os.path as osp
import tqdm
import wandb
import argparse
import torch.distributed as dist
import torchvision
from utils.dataset import dataset_dict
from utils.loggerx import LoggerX
from utils.sampler import RandomSampler
from utils.ops import load_network
import numpy as np


class TrainTask(object):

    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggerX(save_root=osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', '{}_{}'.format(opt.model_name, opt.run_name)))
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.set_loader()
        self.set_model()
        self.gens = []
        self.refs = []

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('Default arguments for training of different methods')

        parser.add_argument('--save_freq', type=int, default=2500,
                            help='save frequency')
        parser.add_argument('--batch_size', type=int, default=1,
                            help='batch_size')
        parser.add_argument('--test_batch_size', type=int, default=1,
                            help='test_batch_size')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='num of workers to use')
        parser.add_argument('--max_iter', type=int, default=150000,
                            help='number of training iterations')
        parser.add_argument('--resume_iter', type=int, default=0,
                            help='number of training epochs')
        parser.add_argument('--test_iter', type=int, default=150000,
                            help='number of epochs for test')
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--mode", type=str, default='train')
        parser.add_argument('--wandb', action="store_true")

        # run_name and model_name
        parser.add_argument('--run_name', type=str, default='default',
                            help='each run name')
        parser.add_argument('--model_name', type=str, default='corediff',
                            help='the type of method')

        # training parameters for one-shot learning framework
        parser.add_argument("--osl_max_iter", type=int, default=3001,
                            help='number of training iterations for one-shot learning framework training')
        parser.add_argument("--osl_batch_size", type=int, default=8,
                            help='batch size for one-shot learning framework training')
        parser.add_argument("--index", type=int, default=10,
                            help='slice index selected for one-shot learning framework training')
        parser.add_argument("--unpair", action="store_true",
                            help='use unpaired data for one-shot learning framework training')
        parser.add_argument("--patch_size", type=int, default=256,
                            help='patch size used to divide the image')

        # dataset
        parser.add_argument('--train_dataset', type=str, default='mayo_2016_sim')
        parser.add_argument('--test_dataset', type=str, default='mayo_2016_sim')   # mayo_2020, piglte, phantom, mayo_2016
        parser.add_argument('--test_id', type=int, default=9,
                            help='test patient index for Mayo 2016')
        parser.add_argument('--context', action="store_true",
                            help='use contextual information')   #
        parser.add_argument('--image_size', type=int, default=128)
        parser.add_argument('--dose', type=int, default=5,
                            help='dose% data use for training and testing')

        return parser

    @staticmethod
    def build_options():
        pass

    def load_pretrained_dict(self, file_name: str):
        self.project_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        return load_network(osp.join(self.project_root, 'pretrained', file_name))

    def set_loader(self):
        opt = self.opt

        if opt.mode == 'train':
            train_dataset = dataset_dict['train'](
                dataset=opt.train_dataset,
                test_id=opt.test_id,
                dose=opt.dose,
                context=opt.context,
            )
            train_sampler = RandomSampler(dataset=train_dataset, batch_size=opt.batch_size,
                                          num_iter=opt.max_iter,
                                          restore_iter=opt.resume_iter)

            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=opt.batch_size,
                sampler=train_sampler,
                shuffle=False,
                drop_last=False,
                num_workers=opt.num_workers,
                pin_memory=True
            )
            self.train_loader = train_loader

        test_dataset = dataset_dict[opt.test_dataset](
            dataset=opt.test_dataset,
            test_id=opt.test_id,
            dose=opt.dose,
            context=opt.context
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True
        )
        self.test_loader = test_loader

        test_images = [test_dataset[i] for i in range(20, min(150, len(test_dataset)), 160)] # 此处影响gen images的batch size，之前300的时候默认是4个
        low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
        full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images], dim=0).cuda()
        self.test_images = (low_dose, full_dose)

        self.test_dataset = test_dataset


    def fit(self):
        opt = self.opt
        if opt.mode == 'train':
            if opt.resume_iter > 0:
                self.logger.load_checkpoints(opt.resume_iter) # j加载参数

            # training routine
            loader = iter(self.train_loader)
            ssim_list =[]
            ssim_admm_list = []
            progress_bar = tqdm.tqdm(range(opt.resume_iter + 1, opt.max_iter + 1), disable=(self.rank != 0), dynamic_ncols=True)
            for n_iter in progress_bar:
                inputs = next(loader)
                loss_visual,lr_visual,gen_full_dose,ref,ssim_diff,ssim_admm,loss_admm = self.train(inputs, n_iter)
                ssim_list.append(ssim_diff)
                ssim_admm_list.append(ssim_admm)
                if n_iter % opt.save_freq == 0:
                    self.logger.checkpoints(n_iter) # 在此处保存模型参数
                    self.test(n_iter)
                    self.generate_images(n_iter)
                    admm_mean_previous = np.mean(np.array(ssim_admm_list))
                    diff_mean_previous = np.mean(np.array(ssim_list))
                    ssim_list = []
                    ssim_list.append(diff_mean_previous)
                    ssim_admm_list = []
                    ssim_admm_list.append(admm_mean_previous)
                progress_bar.set_postfix(


                    loss_diff=loss_visual,
                    loss_admm = loss_admm,
                    # lr=lr_visual,
                    ssim_admm = ssim_admm,
                    ssim_admm_mean =np.mean(np.array(ssim_admm_list)),
                    ssim_diff = ssim_diff,
                    ssim_diff_mean =np.mean(np.array(ssim_list)),

                )
            #     self.gens.append(gen_full_dose)
            #     self.refs.append(ref)
            # print(len(self.gens))
                if opt.wandb:
                    wandb.log({'epoch': n_iter, 'loss_diff': loss_visual,'loss_admm':loss_admm,'ssim_diff_mean':np.mean(np.array(ssim_list)),'ssim_admm_mean':np.mean(np.array(ssim_admm_list))})
        elif opt.mode == 'test':
            self.logger.load_checkpoints(opt.test_iter) #训练模式下在此处加载模型参数
            # self.test(opt.test_iter)
            self.generate_images(opt.test_iter)

        # train one-shot learning framework
        elif opt.mode == 'train_osl_framework':
            self.logger.load_checkpoints(opt.test_iter)
            self.train_osl_framework(opt.test_iter)

        # test one-shot learning framework
        elif opt.mode == 'test_osl_framework':
            self.logger.load_checkpoints(opt.test_iter)
            self.test_osl_framework(opt.test_iter)


    def set_model(opt):
        pass

    def train(self, inputs, n_iter):
        pass

    @torch.no_grad()
    def test(self, n_iter):
        pass

    @torch.no_grad()
    def generate_images(self, n_iter):
        pass

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        pass


    # denormalize to [0, 255] for calculating PSNR, SSIM and RMSE
    # def transfer_calculate_window(self, img, MIN_B=-1024, MAX_B=3072, cut_min=-1000, cut_max=1000):
    #     img = img * (MAX_B - MIN_B) + MIN_B
    #     img[img < cut_min] = cut_min
    #     img[img > cut_max] = cut_max
    #     img = 255 * (img - cut_min) / (cut_max - cut_min)
    #     return img
    def transfer_calculate_window(self,img, MIN_SUVR=0, MAX_SUVR=1):
        # img =  (img - img.min()) / (img.max() - img.min())
        # img = (img - img.min())
        img = torch.clamp(img,0)
        return img

    # # denormalize to [-100, 200]HU for display
    # def transfer_display_window(self, img, MIN_B=-1024, MAX_B=3072, cut_min=-100, cut_max=200):
    #     img = img * (MAX_B - MIN_B) + MIN_B
    #     img[img < cut_min] = cut_min
    #     img[img > cut_max] = cut_max
    #     img = (img - cut_min) / (cut_max - cut_min)
    #     return img

    def transfer_display_window(self,img, MIN_SUVR=0, MAX_SUVR=1, cut_min=0, cut_max=1):
        img = img * (MAX_SUVR - MIN_SUVR) + MIN_SUVR  # Denormalize
        img[img < cut_min] = cut_min  # Apply window
        img[img > cut_max] = cut_max
        img = (img - cut_min) / (cut_max - cut_min)  # Normalize to [0, 1] for display
        return img

