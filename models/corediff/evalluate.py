
import torch
from admm_v3_loop_3L64_dual_diffu import admm
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt
import time

model = admm()
checkpoint = torch.load(r"C:\Users\Administrator\Desktop\admm\admm\eval\12_checkpoint.pth.tar", map_location='cpu')

model.load_state_dict(checkpoint['model'])
model.cuda()
model.eval()
with torch.no_grad():
    img = torch.from_numpy(np.load(r"D:\brainweb_corediff_all\L004_dose080_slice050_sim0_img.npy"))
    img = img.unsqueeze(0)
    img = F.resize(img, (128,128), antialias=True,
                            interpolation=InterpolationMode.BILINEAR).cuda().float().unsqueeze(0)

    cleanx, cleanx_list, noisyx, noisyx_list, diff, noise_upre_list = model(img)
    np.save(r"C:\Users\Administrator\Desktop\noisy.png", noisyx.cpu().squeeze())
    np.save(r"C:\Users\Administrator\Desktop\clean.png", cleanx.cpu().squeeze())



