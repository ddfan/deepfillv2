import os
import math
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

import utils

class InpaintDataset(Dataset):
    def __init__(self, opt, validation=False, test=False):
        self.opt = opt
        self.validation = validation
        self.test = test

        list_IDs = utils.get_jpgs(opt.baseroot)
        n_split = int(self.opt.train_test_split * len(list_IDs))
        list_IDs_train = list_IDs[:n_split]
        train_idxs = np.arange(len(list_IDs_train))
        np.random.shuffle(train_idxs)  # validation set is random subset
        n_train_no_val = int(self.opt.train_val_split * len(list_IDs_train))
        list_IDs_val = [list_IDs_train[idx] for idx in train_idxs[n_train_no_val:]]
        list_IDs_train = [list_IDs_train[idx] for idx in train_idxs[:n_train_no_val]]
        list_IDs_test = list_IDs[n_split:]

        if self.validation:
            self.imglist = list_IDs_val
        elif self.test:
            self.imglist = list_IDs_test
        else:
            self.imglist = list_IDs_train

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        datafilename = self.imglist[index]
        datafilepath = os.path.join(self.opt.baseroot, datafilename)
        data = np.load(datafilepath)

        grayscale = data["num_points"]
        groundtruth = data["risk"]

        #####  Data augmentation ######
        if not self.validation:
            # rot
            rot_rand = np.random.randint(0, 4)
            grayscale = np.rot90(grayscale, k=-rot_rand)
            groundtruth = np.rot90(groundtruth, k=-rot_rand)

            # flip
            flip_rand = np.random.randint(0, 2)
            if flip_rand == 0:  # flip
                grayscale = grayscale[:,::-1]
                groundtruth = groundtruth[:,::-1]

            # add pepper noise
            grayscale = self.add_noise_to_img(grayscale)
            groundtruth = self.add_noise_to_img(groundtruth)

            # random distortion in scale
            scale_rand = np.random.rand() * 0.2 + 0.9
            grayscale = grayscale * scale_rand
            groundtruth = groundtruth * scale_rand
       
        #####################################

        # generate masks
        valid_ground_truth = np.isfinite(groundtruth)
        valid_input = np.isfinite(grayscale)
        # if self.validation:  # generate mask from known mask
        mask = valid_input * valid_ground_truth
        # else:  # generate mask from groundtruth, with random variation
        #     random_mask = self.random_ff_mask(
        #         shape=self.opt.imgsize,
        #         max_angle=self.opt.max_angle,
        #         max_len=self.opt.max_len,
        #         max_width=self.opt.max_width,
        #         times=self.opt.mask_num,
        #     )[0, ...]
        #     mask = random_mask * valid_ground_truth

        # set invalid pixels to 0
        grayscale = np.nan_to_num(grayscale)
        groundtruth = np.nan_to_num(groundtruth)

        if self.opt.view_input_only:
            import matplotlib.pyplot as plt

            plt.subplot(221)
            plt.imshow(grayscale)
            plt.title("num_points")
            # plt.colorbar()
            plt.subplot(222)
            plt.imshow(mask)
            plt.title("mask")
            # plt.colorbar()
            plt.subplot(223)
            plt.imshow(groundtruth)
            plt.title("ground_truth")
            # plt.colorbar()
            plt.subplot(224)
            plt.imshow(valid_ground_truth * 1.0)
            plt.title("output_mask")
            plt.colorbar()
            plt.show()
            quit()

        grayscale = torch.from_numpy(grayscale.astype(np.float32)).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        groundtruth = torch.from_numpy(groundtruth.astype(np.float32)).contiguous()
        output_mask = torch.from_numpy(
            valid_ground_truth.astype(np.float32)
        ).contiguous()

        # grayscale: 1 * 256 * 256; mask: 1 * 256 * 256

        # flatten data for libtorch compatability
        grayscale = torch.flatten(grayscale, start_dim=0)
        mask = torch.flatten(mask, start_dim=0)
        groundtruth = torch.flatten(groundtruth, start_dim=0)
        output_mask = torch.flatten(output_mask, start_dim=0)

        return grayscale, mask, groundtruth, output_mask

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
                (bbox[0] + h) : (bbox[0] + bbox[2] - h),
                (bbox[1] + w) : (bbox[1] + bbox[3] - w),
            ] = 1.0
        return mask.reshape((1,) + mask.shape).astype(np.float32)

    def add_noise_to_img(self, img, mean=0, sigma=0.01):
        gauss = np.random.normal(mean, sigma, img.shape)
        gauss = gauss.reshape(img.shape)
        return img + gauss
