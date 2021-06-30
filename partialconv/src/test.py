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

import scipy.stats

class Tester(object):
    def __init__(self, config, device, model, dataset_test, dataset_test_variance=None):
        self.config = config
        self.device = device
        self.model = model
        self.dataset_test = dataset_test
        self.dataset_test_variance = dataset_test_variance

        self.test_writer = SummaryWriter(self.config.ckpt + "/test", flush_secs=1)
        write_metadata(self.test_writer, self.config)

    def iterate(self):
        
        # # save visualization
        # print('Saving plots...')
        # filename = self.config.ckpt + "/test/" + "sample_img"
        # if self.config.use_cvar_loss:
        #     visualize_cvar(self.model, self.config, self.test_writer, self.device, self.dataset_test, epoch=0, filename=filename)
        # else:
        #     visualize_l1loss(self.model, self.config, self.test_writer, self.device, self.dataset_test, epoch=0, filename=filename)

        # compute statistics
        print('Computing statistics...')
        if self.dataset_test_variance is not None:
            self.compute_statistics_model_based_cvar(self.dataset_test_variance)
        elif self.config.use_cvar_loss:
            self.compute_statistics_cvar(self.dataset_test)
        else:
            self.compute_statistics_l1loss(self.dataset_test)

    def compute_constant_quantiles(self, dataloader):
        # precompute var and cvar, non-model based 
        var_intercept = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        for step, (inputs, mask, gt, alpha) in enumerate(dataloader):
            mask = mask.to(self.device)
            gt = gt.to(self.device)
            for i, alpha_val in enumerate(self.config.alpha_stats_val):
                var_intercept[step,i] = torch.quantile(gt[mask==1], q=alpha_val)
        var_intercept = torch.mean(var_intercept,axis=0)

        cvar_intercept = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        cvar_num_valid = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        for step, (inputs, mask, gt, alpha) in enumerate(dataloader):
            mask = mask.to(self.device)
            gt = gt.to(self.device)
            for i, alpha_val in enumerate(self.config.alpha_stats_val):
                valid_cvar = torch.le(var_intercept[i], gt) * mask
                cvar_intercept[step,i] = torch.sum(valid_cvar * gt)
                cvar_num_valid[step,i] = torch.sum(valid_cvar)
        cvar_intercept = torch.sum(cvar_intercept,axis=0) / torch.sum(cvar_num_valid, axis=0)

        return var_intercept, cvar_intercept

    def plot_stats(self, title="Test Performance", tag="stats", alpha_implied=None, r2_var=None, r2_cvar=None):
        # make statistics plot
        boxplot_width = 0.03
        fig = plt.figure(figsize=(4, 4), dpi=150)

        plt.subplot(311)
        plt.boxplot(alpha_implied, positions=self.config.alpha_stats_val, widths=boxplot_width,
            flierprops={'markerfacecolor': 'k', 'markeredgecolor': 'k', 'marker': '.', 'markersize': 1})
        plt.ylabel(r'$Implied ~\alpha$')
        plt.title(title)
        plt.xlim((min(self.config.alpha_stats_val) - boxplot_width, max(self.config.alpha_stats_val) + boxplot_width))
        plt.ylim((0, 1))
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        plt.subplot(312)
        plt.boxplot(r2_var, positions=self.config.alpha_stats_val, widths=boxplot_width,
            flierprops={'markerfacecolor': 'k', 'markeredgecolor': 'k', 'marker': '.', 'markersize': 1})
        plt.ylabel(r"$VaR~R^2$")
        plt.xlim((min(self.config.alpha_stats_val) - boxplot_width, max(self.config.alpha_stats_val) + boxplot_width))
        plt.ylim((0, 1.0))
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        plt.subplot(313)
        plt.boxplot(r2_cvar, positions=self.config.alpha_stats_val, widths=boxplot_width,
            flierprops={'markerfacecolor': 'k', 'markeredgecolor': 'k', 'marker': '.', 'markersize': 1})
        plt.ylabel(r"$CVaR ~ R^2$")
        plt.xlabel(r"$\alpha$")
        plt.xlim((min(self.config.alpha_stats_val) - boxplot_width, max(self.config.alpha_stats_val) + boxplot_width))
        plt.ylim((0, 1.0))
        plt.gca().set_xticklabels([str(alph).lstrip('0') for alph in self.config.alpha_stats_val])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.13)
        plt.savefig(self.config.ckpt + '/test/' + tag + '.pdf', bbox_inches="tight")

        plt.show()

    def compute_statistics_l1loss(self, dataset):
        self.model.eval()

        dataloader = DataLoader(dataset,
                               batch_size=1,
                               shuffle=False)

        var_intercept, cvar_intercept = self.compute_constant_quantiles(dataloader)
        
        # compute statistics for var and cvar estimates
        n_samples = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        n_gt_less_than_var = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        r2_var = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        r2_cvar = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)

        if self.config.show_progress_bar:
            dataloader = tqdm(dataloader, file=sys.stdout)
        model_time = []
        for step, (inputs, mask, gt, alpha) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            mask = mask.to(self.device)
            gt = gt.to(self.device)
            alpha = alpha.to(self.device)            
            mask_unflat = torch.reshape(mask, (-1, 1, self.config.img_size, self.config.img_size))                
            gt_unflat = torch.reshape(gt, (-1, 1, self.config.img_size, self.config.img_size))
            alpha_unflat = torch.reshape(alpha, (-1, 1, self.config.img_size, self.config.img_size))

            for i, alpha_val in enumerate(self.config.alpha_stats_val):
                alpha_test = torch.ones_like(alpha) * alpha_val

                start_time = time.time()
                output = self.model(inputs, mask, alpha_test)
                model_time.append(time.time() - start_time)

                output = torch.reshape(output, (-1, self.config.out_channels, self.config.img_size, self.config.img_size))

                var = output[:,0:1, :, :]
                cvar = output[:,0:1, :, :]

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

        print("l1 loss model query time: ", str(np.mean(model_time)), str(np.std(model_time)), str(len(model_time)))        

        alpha_implied = n_gt_less_than_var / n_samples
        alpha_implied = alpha_implied.detach().cpu().numpy()
        r2_var = r2_var.detach().cpu().numpy()
        r2_cvar = r2_cvar.detach().cpu().numpy()

        self.plot_stats(alpha_implied=alpha_implied, r2_var=r2_var, r2_cvar=r2_cvar)


    def compute_statistics_cvar(self, dataset):
        self.model.eval()

        dataloader = DataLoader(dataset,
                               batch_size=1,
                               shuffle=False)

        var_intercept, cvar_intercept = self.compute_constant_quantiles(dataloader)

        # compute statistics for var and cvar estimates
        n_samples = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        n_gt_less_than_var = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        r2_var = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        r2_cvar = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)

        if self.config.show_progress_bar:
            dataloader = tqdm(dataloader, file=sys.stdout)
        model_time = []
        for step, (inputs, mask, gt, alpha) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            mask = mask.to(self.device)
            gt = gt.to(self.device)
            alpha = alpha.to(self.device)            
            mask_unflat = torch.reshape(mask, (-1, 1, self.config.img_size, self.config.img_size))                
            gt_unflat = torch.reshape(gt, (-1, 1, self.config.img_size, self.config.img_size))
            alpha_unflat = torch.reshape(alpha, (-1, 1, self.config.img_size, self.config.img_size))

            for i, alpha_val in enumerate(self.config.alpha_stats_val):
                alpha_test = torch.ones_like(alpha) * alpha_val

                start_time = time.time()
                with torch.no_grad():
                    output = self.model(inputs, mask, alpha_test)
                model_time.append(time.time() - start_time)

                output = torch.reshape(output, (-1, self.config.out_channels, self.config.img_size, self.config.img_size))

                if self.config.use_cvar_less_var:
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

        print("cvar model query time: ", str(np.mean(model_time)), str(np.std(model_time)), str(len(model_time)))        

        alpha_implied = n_gt_less_than_var / n_samples
        alpha_implied = alpha_implied.detach().cpu().numpy()
        r2_var = r2_var.detach().cpu().numpy()
        r2_cvar = r2_cvar.detach().cpu().numpy()

        self.plot_stats(alpha_implied=alpha_implied, r2_var=r2_var, r2_cvar=r2_cvar)

    def compute_statistics_model_based_cvar(self, dataset):
        dataloader = DataLoader(dataset,
                               batch_size=1,
                               shuffle=False)

        var_intercept, cvar_intercept = self.compute_constant_quantiles(dataloader)

        # compute statistics for var and cvar estimates
        n_samples = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        n_gt_less_than_var = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        r2_var = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)
        r2_cvar = torch.zeros((len(dataloader), len(self.config.alpha_stats_val))).to(self.device)

        if self.config.show_progress_bar:
            dataloader = tqdm(dataloader, file=sys.stdout)
        model_time = []
        for step, (inputs, mask, gt, alpha, variance) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            mask = mask.to(self.device)
            gt = gt.to(self.device)
            alpha = alpha.to(self.device)
            variance = variance.to(self.device)
            mask_unflat = torch.reshape(mask, (-1, 1, config.img_size, config.img_size))
            gt_unflat = torch.reshape(gt, (-1, 1, config.img_size, config.img_size))
            alpha_unflat = torch.reshape(alpha, (-1, 1, config.img_size, config.img_size))
            variance_unflat = torch.reshape(variance, (-1, 1, config.img_size, config.img_size))

            for i, alpha_val in enumerate(config.alpha_stats_val):
                alpha_test = torch.ones_like(alpha) * alpha_val

                # compute model-based var and cvar
                quantile = scipy.stats.norm.ppf(alpha_val)
                var_scale = scipy.stats.norm.pdf(quantile) / (1. - alpha_val)
                cvar = gt_unflat + variance_unflat * var_scale
                var = gt_unflat + quantile * variance_unflat

                n_samples[step, i] = torch.sum(mask_unflat)
                n_gt_less_than_var[step, i] = torch.sum(mask_unflat * torch.lt(gt_unflat, var))

                r2_var_num = torch.sum(mask_unflat * (alpha_val * torch.clamp(gt_unflat - var, min=0) +
                    (1 - alpha_val) * torch.clamp(var - gt_unflat, min=0))).detach()
                r2_var_denom = torch.sum(mask_unflat * (alpha_val * torch.clamp(gt_unflat - var_intercept[i], min=0) +
                    (1 - alpha_val) * torch.clamp(var_intercept[i] - gt_unflat, min=0))).detach()
                r2_var[step, i] = 1.0 - r2_var_num / r2_var_denom
                # print("var:",i, r2_var_num, r2_var_denom)

                valid_cvar_num = torch.le(var, gt_unflat) * mask_unflat
                r2_cvar_num = torch.sum(torch.abs(cvar -
                    (var + valid_cvar_num * (gt_unflat - var) / (1.0 - alpha_val))))
                valid_cvar_denom = torch.le(var_intercept[i], gt_unflat) * mask_unflat
                r2_cvar_denom = torch.sum(torch.abs(cvar_intercept[i] -
                    (var_intercept[i] + valid_cvar_denom * (gt_unflat - var_intercept[i]) / (1.0 - alpha_val))))
                # print("cvar:",i, r2_cvar_num, r2_cvar_denom)
                # if r2_cvar_denom == 0:
                # r2_cvar[step,i] = 1.0 - r2_cvar_num.detach()
                # else:
                r2_cvar[step, i] = 1.0 - r2_cvar_num.detach() / r2_cvar_denom.detach()

                # cvar_mse[step, i] = torch.mean(torch.square((cvar - gt_unflat) * mask_unflat)).detach()

                # r2_var[step,i] = 1.0 - cvar_mse[step,i] * torch.sum(mask_unflat) / \
                # torch.sum(torch.square((gt_unflat - gt_mean) * mask_unflat)).detach()

        alpha_implied = n_gt_less_than_var / n_samples
        alpha_implied = alpha_implied.detach().cpu().numpy()
        r2_var = r2_var.detach().cpu().numpy()
        r2_cvar = r2_cvar.detach().cpu().numpy()

        self.plot_stats(alpha_implied=alpha_implied, r2_var=r2_var, r2_cvar=r2_cvar)
