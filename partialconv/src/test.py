import torch
from torch.utils.data import DataLoader

from .vis import *
from .utils import write_metadata
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import time

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

        # compute statistics for var and cvar estimates
        n_samples = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        n_gt_less_than_var = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        cvar_mse = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        r2_var = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        r2_cvar = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)

        # precompute var and cvar, non-model based 
        var_intercept = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        for step, (inputs, mask, gt, alpha) in enumerate(dataloader):
            mask = mask.to(device)
            gt = gt.to(device)
            for i, alpha_val in enumerate(config.alpha_stats_val):
                var_intercept[step,i] = torch.quantile(gt[mask==1], q=alpha_val)
        var_intercept = torch.mean(var_intercept,axis=0)

        cvar_intercept = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        cvar_num_valid = torch.zeros((len(dataloader), len(config.alpha_stats_val))).to(device)
        for step, (inputs, mask, gt, alpha) in enumerate(dataloader):
            mask = mask.to(device)
            gt = gt.to(device)
            for i, alpha_val in enumerate(config.alpha_stats_val):
                valid_cvar = torch.le(var_intercept[i], gt) * mask
                cvar_intercept[step,i] = torch.sum(valid_cvar * gt)
                cvar_num_valid[step,i] = torch.sum(valid_cvar)
        cvar_intercept = torch.sum(cvar_intercept,axis=0) / torch.sum(cvar_num_valid, axis=0)
        
        # clamp these guys to prevent them from both being 1.0, which makes for NaN stats
        # var_intercept = torch.clamp(var_intercept, max=0.99)
        # cvar_intercept = torch.clamp(cvar_intercept, max=0.99)
        
        if self.config.show_progress_bar:
            dataloader = tqdm(dataloader, file=sys.stdout)
        model_time = []
        for step, (inputs, mask, gt, alpha) in enumerate(dataloader):
            inputs = inputs.to(device)
            mask = mask.to(device)
            gt = gt.to(device)
            alpha = alpha.to(device)            
            mask_unflat = torch.reshape(mask, (-1, 1, config.img_size, config.img_size))                
            gt_unflat = torch.reshape(gt, (-1, 1, config.img_size, config.img_size))
            alpha_unflat = torch.reshape(alpha, (-1, 1, config.img_size, config.img_size))

            for i, alpha_val in enumerate(config.alpha_stats_val):
                alpha_test = torch.ones_like(alpha) * alpha_val

                start_time = time.time()
                output = model(inputs, mask, alpha_test)
                model_time.append(time.time() - start_time)

                output = torch.reshape(output, (-1, config.out_channels, config.img_size, config.img_size))

                if config.use_cvar_less_var:
                    var = output[:,0:1, :, :]
                    cvar = output[:,0:1, :, :] + output[:,1:2, :, :]
                else:
                    var = output[:,0:1, :, :]
                    cvar = output[:,1:2, :, :]

                n_samples[step,i] = torch.sum(mask_unflat)
                n_gt_less_than_var[step,i] = torch.sum(mask_unflat * torch.lt(gt_unflat, var)) 

                r2_var_num = torch.sum(mask_unflat * (alpha_val * torch.clamp(gt_unflat - var, min=0) + \
                    (1-alpha_val) * torch.clamp(var - gt_unflat, min=0))).detach()
                r2_var_denom = torch.sum(mask_unflat * (alpha_val * torch.clamp(gt_unflat - var_intercept[i], min=0) + \
                    (1-alpha_val) * torch.clamp(var_intercept[i] - gt_unflat, min=0))).detach()
                r2_var[step, i] = 1.0 - r2_var_num / r2_var_denom
                # print("var:",i, r2_var_num, r2_var_denom)

                valid_cvar_num = torch.le(var, gt_unflat) * mask_unflat
                r2_cvar_num = torch.sum(torch.abs(cvar - \
                    (var + valid_cvar_num * (gt_unflat - var) / (1.0 - alpha_val))))
                valid_cvar_denom = torch.le(var_intercept[i], gt_unflat) * mask_unflat
                r2_cvar_denom = torch.sum(torch.abs(cvar_intercept[i] - \
                    (var_intercept[i] + valid_cvar_denom * (gt_unflat - var_intercept[i]) / (1.0 - alpha_val))))
                # print("cvar:",i, r2_cvar_num, r2_cvar_denom)
                # if r2_cvar_denom == 0:
                    # r2_cvar[step,i] = 1.0 - r2_cvar_num.detach()
                # else:
                r2_cvar[step, i] = 1.0 - r2_cvar_num.detach() / r2_cvar_denom.detach()
                                
                # cvar_mse[step, i] = torch.mean(torch.square((cvar - gt_unflat) * mask_unflat)).detach()

                # r2_var[step,i] = 1.0 - cvar_mse[step,i] * torch.sum(mask_unflat) / \
                    # torch.sum(torch.square((gt_unflat - gt_mean) * mask_unflat)).detach()

        print("avg model query time: ", str(np.mean(model_time)))
        
        alpha_implied = n_gt_less_than_var / n_samples
        alpha_implied = alpha_implied.detach().cpu().numpy()
        r2_var = r2_var.detach().cpu().numpy()
        r2_cvar = r2_cvar.detach().cpu().numpy()

        # make statistics plot
        boxplot_width = 0.03
        fig = plt.figure(figsize=(4, 4), dpi=150)

        plt.subplot(311)
        plt.boxplot(alpha_implied, positions=config.alpha_stats_val, widths=boxplot_width, 
            flierprops={'markerfacecolor': 'k', 'markeredgecolor':'k', 'marker': '.', 'markersize': 1})
        plt.ylabel(r'$Implied ~\alpha$')
        plt.title("Test Performance")
        plt.xlim((min(config.alpha_stats_val)-boxplot_width, max(config.alpha_stats_val)+boxplot_width))
        plt.ylim((0,1))
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        
        plt.subplot(312)
        plt.boxplot(r2_var, positions=config.alpha_stats_val, widths=boxplot_width,
            flierprops={'markerfacecolor': 'k', 'markeredgecolor':'k', 'marker': '.', 'markersize': 1})
        plt.ylabel(r"$VaR~R^2$")
        plt.xlim((min(config.alpha_stats_val)-boxplot_width, max(config.alpha_stats_val)+boxplot_width))
        plt.ylim((0,1.0))
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

        plt.subplot(313)
        plt.boxplot(r2_cvar, positions=config.alpha_stats_val, widths=boxplot_width,
            flierprops={'markerfacecolor': 'k', 'markeredgecolor':'k', 'marker': '.', 'markersize': 1})
        plt.ylabel(r"$CVaR ~ R^2$")
        plt.xlabel(r"$\alpha$")
        plt.xlim((min(config.alpha_stats_val)-boxplot_width, max(config.alpha_stats_val)+boxplot_width))
        plt.ylim((0, 1.0))
        plt.gca().set_xticklabels([str(alph).lstrip('0') for alph in config.alpha_stats_val])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.13)
        plt.savefig(self.config.ckpt + '/test/stats.pdf', bbox_inches="tight")

        plt.show()
