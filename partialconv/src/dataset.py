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

class Places2(Dataset):
    def __init__(self, data_root, img_transform, mask_transform, data='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # get the list of image paths
        if data == 'train':
            self.paths = glob('{}/data_256/**/*.jpg'.format(data_root),
                              recursive=True)
            self.mask_paths = glob('{}/mask/*.png'.format(data_root))
        else:
            self.paths = glob('{}/val_256/*.jpg'.format(data_root, data))
            self.mask_paths = glob('{}/val_mask/*.png'.format(data_root))

        self.N_mask = len(self.mask_paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self._load_img(self.paths[index])
        img = self.img_transform(img.convert('RGB'))
        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return img * mask, mask, img

    def _load_img(self, path):
        """
        For dealing with the error of loading image which is occured by the loaded image has no data.
        """
        try:
            img = Image.open(path)
        except:
            extension = path.split('.')[-1]
            for i in range(10):
                new_path = path.split('.')[0][:-1] + str(i) + '.' + extension
                try:
                    img = Image.open(new_path)
                    break
                except:
                    continue
        return img


class InpaintDataset(Dataset):
    def __init__(self, config, validation=False, test=False):
        self.config = config
        self.validation = validation
        self.test = test

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
        datafilename = self.imglist[index]
        datafilepath = os.path.join(self.config.data_root, datafilename)
        data = np.load(datafilepath)

        input_img = []
        for layer in self.map_layers:
            input_img.append(data[layer])
        input_img = np.stack(input_img, axis=-1)
        groundtruth = data[self.output_layer]

        #####  Data augmentation ######
        if not self.validation:
            # rot
            rot_rand = np.random.randint(0, 4)
            input_img = np.rot90(input_img, k=-rot_rand)
            groundtruth = np.rot90(groundtruth, k=-rot_rand)

            # flip
            flip_rand = np.random.randint(0, 2)
            if flip_rand == 0:  # flip
                input_img = input_img[:,::-1,:]
                groundtruth = groundtruth[:,::-1]

            # # add pepper noise
            # input_img = self.add_noise_to_img(input_img)
            # groundtruth = self.add_noise_to_img(groundtruth)

            # # random distortion in scale
            # scale_rand = np.random.rand() * 0.2 + 0.9
            # input_img = input_img * scale_rand
            # groundtruth = groundtruth * scale_rand
       
            # shift

            # add noise

        #####################################

        # generate masks
        valid_ground_truth = np.isfinite(groundtruth)
        valid_input = np.isfinite(input_img[:,:,0])
        # if self.validation:  # generate mask from known mask
        mask = valid_input * valid_ground_truth
        # else:  # generate mask from groundtruth, with random variation
        #     random_mask = self.random_ff_mask(
        #         shape=self.config.imgsize,
        #         max_angle=self.config.max_angle,
        #         max_len=self.config.max_len,
        #         max_width=self.config.max_width,
        #         times=self.config.mask_num,
        #     )[0, ...]
        #     mask = random_mask * valid_ground_truth

        # set invalid pixels to 0
        input_img = np.nan_to_num(input_img)
        groundtruth = np.nan_to_num(groundtruth)

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(nrows = 1, ncols = 10)
        # for i in range(10):
        #     im = axs[i].imshow(input_img[:,:,i])
        #     fig.colorbar(im, ax=axs[i])
        # plt.show()
        # quit()

        input_img = np.transpose(input_img, axes=(2,0,1))
        input_img = torch.from_numpy(input_img.astype(np.float32)).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        groundtruth = torch.from_numpy(groundtruth.astype(np.float32)).contiguous()

        # input_img: in_channels * 256 * 256; mask: in_channels * 256 * 256

        # flatten data for libtorch compatability
        input_img = torch.flatten(input_img, start_dim=0)
        mask = torch.flatten(mask, start_dim=0)
        groundtruth = torch.flatten(groundtruth, start_dim=0)

        return input_img, mask, groundtruth

    def random_ff_mask(self, shape, max_angle=4, max_len=40, max_width=10, times=15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1,) + mask.shape).astype(np.float32)

    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low=ver_margin, high=maxt)
        l = np.random.randint(low=hor_margin, high=maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[
                (bbox[0] + h): (bbox[0] + bbox[2] - h),
                (bbox[1] + w): (bbox[1] + bbox[3] - w),
            ] = 1.0
        return mask.reshape((1,) + mask.shape).astype(np.float32)

    def add_noise_to_img(self, img, mean=0, sigma=0.01):
        gauss = np.random.normal(mean, sigma, img.shape)
        gauss = gauss.reshape(img.shape)
        return img + gauss
