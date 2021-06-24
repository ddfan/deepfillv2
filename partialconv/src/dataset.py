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
from .voronoi import RandomVoronoiMap

class AddCustomNoise(object):
    def __init__(self, mean=0., std=1., pepper_noise=0.05, gauss_idxs=[], pepper_idxs=[]):
        self.std = std
        self.mean = mean
        self.pepper_noise = pepper_noise
        self.gauss_idxs = gauss_idxs
        self.pepper_idxs = pepper_idxs
        
    def __call__(self, tensor):
        tensor[self.gauss_idxs,...] += torch.randn((len(self.gauss_idxs),tensor.size()[1], tensor.size()[2])) * self.std + self.mean
        tensor[self.pepper_idxs,...] += torch.lt(torch.rand((len(self.pepper_idxs),tensor.size()[1], tensor.size()[2])), self.pepper_noise)
        return tensor 

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, pepper_noise={2}, )'.format(self.mean, self.std, self.pepper_noise)

class CostmapDataset(Dataset):
    def __init__(self, config, validation=False, test=False):

        np.random.seed(0)

        self.config = config
        self.validation = validation
        self.test = test

        self.img_transform = transforms.Compose([
                    transforms.RandomAffine(degrees=(0,360),
                                            translate=(0.0,0.3),
                                            scale=(0.9,1.0),
                                            shear=(-10,10,-10,10)),
                    transforms.RandomVerticalFlip(),
                    # AddCustomNoise(0,0.02,0.01,[1,2],[0,3,4,5,6,7]),
                    ])
        self.voronoi_map = RandomVoronoiMap(img_size=self.config.img_size)

        list_IDs = get_jpgs(config.data_root)
        np.random.shuffle(list_IDs)  # randomly shuffle all the data
        train_end = int(config.train_val_test[0] * len(list_IDs))
        val_end = train_end + int(config.train_val_test[1] * len(list_IDs))
        list_IDs_train = list_IDs[:train_end]
        list_IDs_val = list_IDs[train_end:val_end]
        list_IDs_test = list_IDs[val_end:]
        
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
        alpha = self.voronoi_map.get_random_map()
        alpha = np.expand_dims(alpha, axis=0)
        # alpha = np.ones((1, self.config.img_size, self.config.img_size)) * np.random.rand()
        alpha = alpha * 0.998 + 0.001  # prevent 0 and 1 for numeric stability

        #####  Data augmentation ######
        n_layers = len(self.map_layers)
        if self.validation or self.test:
            input_img = torch.from_numpy(input_img.astype(np.float32)).contiguous()
            mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
            groundtruth = torch.from_numpy(groundtruth.astype(np.float32)).contiguous()
            alpha = torch.from_numpy(alpha.astype(np.float32)).contiguous()
        else:
            img_mask_gt = np.concatenate([input_img, mask, groundtruth, alpha], axis=0)
            img_mask_gt = torch.from_numpy(img_mask_gt.astype(np.float32)).contiguous()
            img_mask_gt_tf = self.img_transform(img_mask_gt)
            input_img = img_mask_gt_tf[:n_layers,:,:]
            mask = img_mask_gt_tf[n_layers:n_layers+1,:,:]
            groundtruth = img_mask_gt_tf[n_layers+1:n_layers+2,:,:]
            alpha = img_mask_gt_tf[n_layers+2:n_layers+3,:,:]

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
