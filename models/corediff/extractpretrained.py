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
from models.corediff.corediff_wrapper import Network, WeightNet
from models.corediff.diffusion_modules import Diffusion
import datetime
import wandb
from pytorch_msssim import SSIM
import time
from models.corediff.admm_v3 import admm
from torch_radon import ParallelBeam



denoise_fn = Network(in_channels=1, context=1)
admmnet = admm()
model = Diffusion(
    denoise_fn=denoise_fn,
    admmnet=None,
    image_size=128,  ## 注意这个地方即使加了--image_size=256 也要改
    timesteps=20,
    context=True
).cpu()

pthfile = r'C:\Users\Administrator\Desktop\CoreDiff-main\output\corediff_forzuhui\save_models\model-144000'  #faster_rcnn_ckpt.pth
pretrained_dict= torch.load(pthfile,map_location=torch.device('cpu'))

unet_dict = model.state_dict()

unet_pre_dict = {k: v for k, v in pretrained_dict.items() if k in unet_dict}

torch.save(unet_pre_dict,r'C:\Users\Administrator\Desktop\CoreDiff-main\output\corediff_forzuhui\save_models\denoise_fn-144000')
print("done")

