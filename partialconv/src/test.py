import torch
from torch.utils.data import DataLoader

from .vis import *
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter


class Tester(object):
    def __init__(self, config, device, model, dataset_test):
        self.config = config
        self.device = device
        self.model = model
        self.dataset_test = dataset_test

        self.test_writer = SummaryWriter(self.config.ckpt + "/test", flush_secs=1)

    def iterate(self):
        print('Start the testing...')
        first_iteration = True

        # save visualization
        if self.config.use_cvar_loss:
            visualize_cvar(self.model, self.config, self.test_writer, self.device, self.dataset_test, epoch=0)
        else:
            visualize_l1loss(self.model, self.config, self.test_writer, self.device, self.dataset_test, epoch=0)

        # compute statistics
        self.compute_statistics_cvar(self.model, self.config, self.test_writer, self.device, self.dataset_test)

    def compute_statistics_cvar(self, model, config, writer, device, dataset):
        model.eval()
        for i in range(len(dataset)):
            inputs, mask, gt, alpha = dataset[i, True] 
            
            outputs = []
            for alpha_val in config.alpha_test_val:
                alpha_test = torch.ones_like(alpha) * alpha_val
                output = model(inputs.to(device), mask.to(device), alpha_test.to(device))
                output = output.to(torch.device('cpu'))
                output = torch.reshape(output, (-1, config.out_channels, config.img_size, config.img_size))
                outputs.append(output)

            # unflatten images
            inputs = torch.reshape(inputs, (-1, config.in_channels, config.img_size, config.img_size))
            mask = torch.reshape(mask, (-1, 1, config.img_size, config.img_size))
            gt = torch.reshape(gt, (-1, 1, config.img_size, config.img_size))

            # assemble var and cvar images
            vars = []
            cvars = []
            for i in range(len(config.alpha_test_val)):
                if config.use_cvar_less_var:
                    var = outputs[i][:, 1:2, :, :]
                    cvar = outputs[i][:, 0:1, :, :] + outputs[i][:, 1:2, :, :]
                else:
                    var = outputs[i][:, 0:1, :, :]
                    cvar = outputs[i][:, 1:2, :, :]
                vars.append(var * mask)
                cvars.append(cvar * mask)
            vars = torch.cat(vars, dim=1)
            cvars = torch.cat(cvars, dim=1)

            print(vars.size())
