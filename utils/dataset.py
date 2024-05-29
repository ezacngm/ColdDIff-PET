import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
import yaml
from utils.mypath import MyPath
from functools import partial
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import cv2
class CTDataset(Dataset):
    def __init__(self, dataset, mode, test_id=9, dose=5, context=False):
        self.mode = mode
        self.context = context
        print(dataset)
        data_gt_root =  r'D:\ppet_corediff'
        if dataset in ['mayo_2016_sim', 'mayo_2016']:
            if dataset == 'mayo_2016_sim':
                data_root = r'D:\ppet_corediff'
            elif dataset == 'mayo_2016':
                data_root = r'D:\ppet_corediff'


            patient_ids = list(range(0,25))
            # patient_ids = [4, 5, 6, 18, 20, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
            # patient_ids = [4, 5, 6, 18, 20, 38, 41, 42, 43, 44]
            if mode == 'train':
                patient_ids.pop(test_id)
            elif mode == 'test':
                patient_ids = patient_ids[test_id:test_id + 1]

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_gt_root, (r'L{:03d}_dose100_'.format(id) + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):

                patient_list = sorted(glob(osp.join(data_root, (r'L{:03d}_dose{:03d}_'.format(id, dose) + '*_img.npy'))))
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
            base_input = patient_lists
        elif dataset == 'mayo_2020':
            data_root = './data_preprocess/gen_data/mayo_2020_npy'
            if dose == 10:
                patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050']
            elif dose == 25:
                patient_ids = ['L077', 'L056', 'L186', 'L006', 'L148']

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, (id + '_target_' + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, (id + '_{}_'.format(dose) + '*_img.npy'))))
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
                base_input = patient_lists


        elif dataset == 'piglet':
            data_root = './data_preprocess/gen_data/piglet_npy'

            patient_list = sorted(glob(osp.join(data_root, 'piglet_target_' + '*_img.npy')))
            base_target = patient_list[1:len(patient_list) - 1]

            patient_list = sorted(glob(osp.join(data_root, 'piglet_{}_'.format(dose) + '*_img.npy')))
            if context:
                cat_patient_list = []
                for i in range(1, len(patient_list) - 1):
                    patient_path = ''
                    for j in range(-1, 2):
                        patient_path = patient_path + '~' + patient_list[i + j]
                    cat_patient_list.append(patient_path)
                    base_input = cat_patient_list
            else:
                patient_list = patient_list[1:len(patient_list) - 1]
                base_input = patient_list


        elif dataset == 'phantom':
            data_root = './data_preprocess/gen_data/xnat_npy'

            patient_list = sorted(glob(osp.join(data_root, 'xnat_target' + '*_img.npy')))[9:21]
            base_target = patient_list[1:len(patient_list) - 1]

            patient_list = sorted(glob(osp.join(data_root, 'xnat_{:0>3d}_'.format(dose) + '*_img.npy')))[9:21]
            if context:
                cat_patient_list = []
                for i in range(1, len(patient_list) - 1):
                    patient_path = ''
                    for j in range(-1, 2):
                        patient_path = patient_path + '~' + patient_list[i + j]
                    cat_patient_list.append(patient_path)
                    base_input = cat_patient_list
            else:
                patient_list = patient_list[1:len(patient_list) - 1]
                base_input = patient_list

        self.input = base_input
        self.target = base_target
        print(len(self.input))
        print(len(self.target))



    def __getitem__(self, index):
        input, target = self.input[index], self.target[index]
        if self.context:
            input = input.split('~')
            inputs = []
            for i in range(1, len(input)):
                inputs.append(np.load(input[i])[np.newaxis, ...].astype(np.float32))
            input = np.concatenate(inputs, axis=0)  #(3, 128, 128)
        else:
            input = np.load(input)
            input = cv2.resize(input, (128, 128), interpolation=cv2.INTER_AREA)
            input = input[np.newaxis, ...].astype(np.float32) #(1, 128, 128)


        target = np.load(target)
        target = cv2.resize(target, (128, 128), interpolation=cv2.INTER_AREA)
        target =target[np.newaxis,...].astype(np.float32) #(1, 128, 128)

        input = self.normalize_(input)
        target = self.normalize_(target)
        return input, target

    def __len__(self):
        return len(self.target)

    # def normalize_(self, img, MIN_B=-1024, MAX_B=3072):
    #     img = img - 1024
    #     img[img < MIN_B] = MIN_B
    #     img[img > MAX_B] = MAX_B
    #     img = (img - MIN_B) / (MAX_B - MIN_B)
    #     return img
    def normalize_(self, img, min_val=None, max_val=None):
        min_val = img.min()
        max_val = img.max()
        # Normalize PET images to a [0, 1] range
        img_normalized = (img - min_val) / (max_val - min_val)
        # Clip values just in case to ensure they remain within [0, 1]
        img_normalized = img_normalized.clip(0, 1)
        # img = transforms.RandomRotation(degrees=(-15, 15))(img)  # todo: 看torch文档 要求输入shape是 BCWH
        # img = transforms.RandomHorizontalFlip(p=0.5)(img)
        return img_normalized

dataset_dict = {
    'train': partial(CTDataset, dataset='mayo_2016_sim', mode='train', test_id=9, dose=5, context=False),
    'mayo_2016_sim': partial(CTDataset, dataset='mayo_2016_sim', mode='test', test_id=9, dose=5, context=False),
    'mayo_2016': partial(CTDataset, dataset='mayo_2016', mode='test', test_id=9, dose=25, context=True),
    'mayo_2020': partial(CTDataset, dataset='mayo_2020', mode='test', test_id=None, dose=None, context=True),
    'piglet': partial(CTDataset, dataset='piglet', mode='test', test_id=None, dose=None, context=True),
    'phantom': partial(CTDataset, dataset='phantom', mode='test', test_id=None, dose=108, context=True),
}
