from glob import glob
import random
import os
import math
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np
import cv2
from PIL import Image
from .utils import get_jpgs
from scipy.ndimage.filters import gaussian_filter

class CostmapDataset(Dataset):
    def __init__(self, config, validation=False, test=False):
        self.config = config
        self.validation = validation
        self.test = test

        self.img_transform = transforms.Compose([
                    transforms.RandomAffine(degrees=(0,360),
                                            translate=(0.0,0.2),
                                            scale=(0.9,1.0)),
                    transforms.RandomVerticalFlip()
                    ])

        list_IDs = get_jpgs(config.data_root)
        n_split = int(self.config.train_test_split * len(list_IDs))
        list_IDs_train = list_IDs[:n_split]
        train_idxs = np.arange(len(list_IDs_train))
        np.random.shuffle(train_idxs)  # validation set is random subset
        n_train_no_val = int(self.config.train_val_split * len(list_IDs_train))
        list_IDs_val = [list_IDs_train[idx] for idx in train_idxs[n_train_no_val:]]
        list_IDs_train = [list_IDs_train[idx] for idx in train_idxs[:n_train_no_val]]
        list_IDs_test = list_IDs[n_split:]

        if self.validation:
            self.imglist = list_IDs_val
        elif self.test:
            self.imglist = list_IDs_test
        else:
            self.imglist = list_IDs_train

        self.map_layers = config.input_map_layers
        self.output_layer = config.output_layer

    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self, index):
        datafilepath = self.imglist[index]
        data = np.load(datafilepath)

        input_img = []
        for layer in self.map_layers:
            input_img.append(data[layer])
        input_img = np.stack(input_img, axis=-1)
        groundtruth = data[self.output_layer]

        # generate masks
        valid_ground_truth = np.isfinite(groundtruth)
        valid_input = np.isfinite(input_img[:,:,0])
        mask = valid_input * valid_ground_truth

        # set invalid pixels to 0
        input_img = np.nan_to_num(input_img)
        groundtruth = np.nan_to_num(groundtruth)

        input_img = np.transpose(input_img, axes=(2,0,1)) # C x H x W
        mask = np.expand_dims(mask, axis=0)
        groundtruth = np.expand_dims(groundtruth, axis=0)

        ####  Set Alpha ######
        # create random image of alphas
        alpha = np.random.normal(0, 1, (1, self.config.img_size, self.config.img_size))
        alpha = gaussian_filter(alpha, sigma=self.config.alpha_random_variance)
        alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
        alpha = alpha * 0.98 + 0.01  # prevent 0 and 1 for numeric stability

        #####  Data augmentation ######
        n_layers = len(self.map_layers)
        if not self.validation:
            img_mask_gt = np.concatenate([input_img, mask, groundtruth, alpha], axis=0)
            img_mask_gt = torch.from_numpy(img_mask_gt.astype(np.float32)).contiguous()
            img_mask_gt_tf = self.img_transform(img_mask_gt)
            input_img = img_mask_gt_tf[:n_layers,:,:]
            mask = img_mask_gt_tf[n_layers:n_layers+1,:,:]
            groundtruth = img_mask_gt_tf[n_layers+1:n_layers+2,:,:]
            alpha = img_mask_gt_tf[n_layers+2:n_layers+3,:,:]
        else:
            input_img = torch.from_numpy(input_img.astype(np.float32)).contiguous()
            mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
            groundtruth = torch.from_numpy(groundtruth.astype(np.float32)).contiguous()
            alpha = torch.from_numpy(alpha.astype(np.float32)).contiguous()

        #####################################

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(nrows = 2, ncols = int((n_layers + 2) / 2.0 + 1.0))
        # axs = axs.flatten()
        # for i in range(n_layers):
        #     im = axs[i].imshow(input_img[i,:,:])
        #     fig.colorbar(im, ax=axs[i])
        # im = axs[n_layers].imshow(mask[0,:,:])
        # fig.colorbar(im, ax=axs[n_layers])
        # im = axs[n_layers+1].imshow(groundtruth[0,:,:])
        # fig.colorbar(im, ax=axs[n_layers])
        # plt.show()
        # quit()

        # flatten data for libtorch compatability
        input_img = torch.flatten(input_img, start_dim=0)
        mask = torch.flatten(mask, start_dim=0)
        groundtruth = torch.flatten(groundtruth, start_dim=0)
        alpha = torch.flatten(alpha, start_dim=0)        

        return input_img, mask, groundtruth, alpha

    def clean_data(self):
        for filename in self.imglist:
            try:
                data = np.load(filename)
            except:
                print(filename)

    def add_noise_to_img(self, img, mean=0, sigma=0.01):
        gauss = np.random.normal(mean, sigma, img.shape)
        gauss = gauss.reshape(img.shape)
        return img + gauss
