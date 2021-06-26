import torch
from torch.utils.data import DataLoader

from .vis import *
from .utils import write_metadata
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Tester(object):
    def __init__(self, config, device, model, dataset_test):
        self.config = config
        self.device = device
        self.model = model
        self.dataset_test = dataset_test

        self.test_writer = SummaryWriter(self.config.ckpt + "/test", flush_secs=1)
        write_metadata(self.test_writer, self.config)

    def iterate(self):
        
        # # save visualization
        # print('Saving plots...')
        # if self.config.use_cvar_loss:
        #     visualize_cvar(self.model, self.config, self.test_writer, self.device, self.dataset_test, epoch=0)
        # else:
        #     visualize_l1loss(self.model, self.config, self.test_writer, self.device, self.dataset_test, epoch=0)

        # compute statistics
        print('Computing statistics...')
        self.compute_statistics_cvar(self.model, self.config, self.test_writer, self.device, self.dataset_test)

    def compute_statistics_cvar(self, model, config, writer, device, dataset):
        model.eval()

        dataloader = DataLoader(dataset,
                               batch_size=1,
                               shuffle=False)
        if self.config.show_progress_bar:
            dataloader = tqdm(dataloader, file=sys.stdout)

        # compute statistics for var and cvar estimates
        n_samples = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        n_gt_less_than_var = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        cvar_mse = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        r2_vals = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        for step, (inputs, mask, gt, alpha) in enumerate(dataloader):
            inputs = inputs.to(device)
            mask = mask.to(device)
            gt = gt.to(device)
            alpha = alpha.to(device)            
            mask_unflat = torch.reshape(mask, (-1, 1, config.img_size, config.img_size))                
            gt_unflat = torch.reshape(gt, (-1, 1, config.img_size, config.img_size))

            for i, alpha_val in enumerate(config.alpha_stats_val):
                alpha_test = torch.ones_like(alpha) * alpha_val
                output = model(inputs, mask, alpha_test)
                output = torch.reshape(output, (-1, config.out_channels, config.img_size, config.img_size))

                if config.use_cvar_less_var:
                    var = output[:,0:1, :, :]
                    cvar = output[:,0:1, :, :] + output[:,1:2, :, :]
                else:
                    var = output[:,0:1, :, :]
                    cvar = output[:,1:2, :, :]

                n_samples[step,i] = torch.sum(mask_unflat)
                n_gt_less_than_var[step,i] = torch.sum(mask_unflat * torch.lt(gt_unflat, var)) 

                cvar_mse[step, i] = torch.mean(torch.square((cvar - gt_unflat) * mask_unflat)).detach()
                gt_mean = torch.mean(gt_unflat * mask_unflat)
                r2_vals[step,i] = 1.0 - cvar_mse[step,i] * torch.sum(mask_unflat) / \
                    torch.sum(torch.square((gt_unflat - gt_mean) * mask_unflat)).detach()

        alpha_implied = n_gt_less_than_var / n_samples
        alpha_implied = alpha_implied.detach().cpu().numpy()
        cvar_mse = cvar_mse.detach().cpu().numpy()
        r2_vals = r2_vals.detach().cpu().numpy()


        plt.subplot(311)
        plt.violinplot(alpha_implied, positions=config.alpha_stats_val, widths=0.05)
        plt.ylabel("implied fraction")
        plt.subplot(312)
        plt.violinplot(cvar_mse, positions=config.alpha_stats_val, widths=0.05)
        plt.ylabel("cvar mse")
        plt.subplot(313)
        plt.violinplot(r2_vals, positions=config.alpha_stats_val, widths=0.05)
        plt.ylabel("r2")
        plt.xlabel("alpha")
        plt.show()
