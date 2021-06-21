import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from .utils import save_ckpt, to_items
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

class Trainer(object):
    def __init__(self, epoch, config, device, model, dataset_train,
                 dataset_val, criterion, optimizer):
        self.epoch = epoch
        self.config = config
        self.device = device
        self.model = model
        self.dataloader_train = DataLoader(dataset_train,
                                       batch_size=config.batch_size,
                                       shuffle=True)
        self.steps = len(self.dataloader_train)
        self.dataset_val = dataset_val
        self.dataloader_val = DataLoader(dataset_val,
                                       batch_size=config.batch_size,
                                       shuffle=False)
        self.criterion = criterion
        self.optimizer = optimizer

        self.train_writer = SummaryWriter(self.config.ckpt + "/train", flush_secs=1)
        self.val_writer = SummaryWriter(self.config.ckpt + "/val", flush_secs=1)

    def iterate(self):
        print('Start the training...')
        first_iteration = True
        best_val_loss = float("inf")
        for i in range(self.config.max_epochs):
            if self.config.show_progress_bar:
                dataloader = tqdm(self.dataloader_train, file=sys.stdout)
            else:
                dataloader = self.dataloader_train

            # train network    
            train_loss_dict = {}
            for step, (input, mask, gt, alpha) in enumerate(dataloader):
                loss_dict = self.evaluate(input, mask, gt, alpha=alpha, add_graph=first_iteration) 
                train_loss_dict = self.accumulate_loss(train_loss_dict, loss_dict, len(dataloader))               

                # report the training loss
                if self.config.print_step_losses and step < self.steps - 1:
                    self.report(self.epoch, step, loss_dict)

                if first_iteration:
                    first_iteration = False

            # log losses to tensorboard
            self.write_losses(train_loss_dict, self.epoch, "train")

            # determine validation loss
            val_loss_dict = {}
            for step, (input, mask, gt, alpha) in enumerate(self.dataloader_val):
                loss_dict = self.evaluate(input, mask, gt, alpha=alpha, train=False)
                val_loss_dict = self.accumulate_loss(val_loss_dict, loss_dict, len(self.dataloader_val))

            self.write_losses(val_loss_dict, self.epoch, "val")

            self.report(self.epoch, 0, val_loss_dict)

            # save visualization
            if self.config.use_cvar_loss:
                self.visualize_cvar(self.dataset_val, epoch = self.epoch)
            else:
                self.visualize(self.dataset_val, epoch = self.epoch)

            # save the model
            new_val_loss = val_loss_dict[self.config.save_model_loss]
            if new_val_loss < best_val_loss or i + 1 == self.config.max_epochs:
                if best_val_loss != float("inf"):
                    print('Model loss improved from {:.6f} to {:.6f}, saving.'.format(best_val_loss, new_val_loss))
                    save_ckpt('{}/models/model_{:04d}'.format(self.config.ckpt,
                                                        self.epoch),
                              [('model', self.model)],
                              [('optimizer', self.optimizer)],
                              self.epoch,
                              self.config)
                best_val_loss = new_val_loss
                
            self.epoch += 1

    def evaluate(self, input, mask, gt, alpha=None, train=True, add_graph=False):
        if train:
            # set the model to training mode
            self.model.train()
        else:
            # set the model to evaluation mode
            self.model.eval()

        # send the input tensors to cuda
        input = input.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)
        if alpha is not None:
            alpha = alpha.to(self.device)

        # model forward
        output = self.model(input, mask, alpha)
        loss_dict = self.criterion(input, mask, output, gt, alpha)
        loss = 0.0
        for key, val in loss_dict.items():
            coef = getattr(self.config, '{}_coef'.format(key))
            loss += coef * val

        if train:
            # updates the model's params
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if add_graph:
            self.val_writer.add_graph(self.model, (input, mask, alpha))

        loss_dict['total'] = loss
        return to_items(loss_dict)

    def visualize_l1loss(self, dataset, filename=None, epoch=0):
        self.model.eval()
        image, mask, gt, _ = zip(*[dataset[i] for i in range(min(len(dataset), self.config.num_vis_imgs))])
        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)

        with torch.no_grad():
            output = self.model(image.to(self.device), mask.to(self.device))
        output = output.to(torch.device('cpu'))

        # unflatten images
        image = torch.reshape(image, (-1, self.config.in_channels, self.config.img_size, self.config.img_size))
        mask = torch.reshape(mask, (-1, 1, self.config.img_size, self.config.img_size))
        output = torch.reshape(output, (-1, self.config.out_channels, self.config.img_size, self.config.img_size))
        gt = torch.reshape(gt, (-1, 1, self.config.img_size, self.config.img_size))

        idx = self.config.input_map_layers.index("obstacle_occupancy")
        mae = torch.abs((gt - output) * mask)
        grid = make_grid(torch.cat([image[:, idx:idx + 1, :, :], output, gt, mae], dim=0), scale_each=True)

        self.val_writer.add_image('images', grid, epoch)
        
        if filename is not None:
            save_image(grid, filename)

    def visualize_cvar(self, dataset, filename=None, epoch=0):
        self.model.eval()
        inputs, mask, gt, alpha = zip(*[dataset[i, True] for i in range(min(len(dataset), self.config.num_vis_imgs))])
        inputs = torch.stack(inputs)
        mask = torch.stack(mask)
        gt = torch.stack(gt)
        alpha = torch.stack(alpha)

        outputs = []
        with torch.no_grad():
            for alpha_val in self.config.alpha_test_val:
                alpha_test = torch.ones_like(alpha) * alpha_val
                output = self.model(inputs.to(self.device), mask.to(self.device), alpha_test.to(self.device))
                output = output.to(torch.device('cpu'))
                output = torch.reshape(output, (-1, self.config.out_channels, self.config.img_size, self.config.img_size))
                outputs.append(output)

            # also query the given alpha from the dataset
            output = self.model(inputs.to(self.device), mask.to(self.device), alpha.to(self.device))
            output = output.to(torch.device('cpu'))
            output = torch.reshape(output, (-1, self.config.out_channels, self.config.img_size, self.config.img_size))
            outputs.append(output)

        # unflatten images
        inputs = torch.reshape(inputs, (-1, self.config.in_channels, self.config.img_size, self.config.img_size))
        mask = torch.reshape(mask, (-1, 1, self.config.img_size, self.config.img_size))
        gt = torch.reshape(gt, (-1, 1, self.config.img_size, self.config.img_size))

        # assemble var and cvar images
        vars = []
        cvars = []
        for i in range(len(self.config.alpha_test_val)):
            if self.config.use_cvar_less_var:
                var = outputs[i][:, 1:2, :, :]
                cvar = outputs[i][:, 0:1, :, :] + outputs[i][:, 1:2, :, :]
            else:
                var = outputs[i][:, 0:1, :, :]
                cvar = outputs[i][:, 1:2, :, :]
            vars.append(var)
            cvars.append(cvar)
        vars = torch.cat(vars, dim=1)
        cvars = torch.cat(cvars, dim=1)

        # assemble varying cvar
        # alpha = torch.reshape(alpha, (-1, 1, self.config.img_size, self.config.img_size))
        if self.config.use_cvar_less_var:
            varying_cvar = outputs[-1][:,0:1, :, :] + outputs[-1][:, 1:2, :, :]
        else:
            varying_cvar = outputs[-1][:,1:2, :, :]
        img_arr = torch.cat([ gt, vars, cvars, varying_cvar], dim=1)

        # create matplotlib figure
        img_arr_np = img_arr.cpu().detach().numpy()
        f, ax = plt.subplots(img_arr_np.shape[0], img_arr_np.shape[1],
            figsize=(img_arr_np.shape[1] * 2, img_arr_np.shape[0] * 2))
        f.tight_layout()
        for i in range(img_arr_np.shape[0]):
            for j in range(img_arr_np.shape[1]):
                ax[i, j].imshow(img_arr_np[i, j, :, :], vmin=0, vmax=1)
                ax[i, j].axis('off')
                ax[i, j].set_aspect('equal')
        f.subplots_adjust(wspace=0, hspace=0)

        self.val_writer.add_figure('output', f, epoch)

        # create input figure
        inputs_np = inputs.cpu().detach().numpy()
        f, ax = plt.subplots(inputs_np.shape[0], inputs_np.shape[1],
            figsize=(inputs_np.shape[1] * 2, inputs_np.shape[0] * 2))
        f.tight_layout()
        for i in range(inputs_np.shape[0]):
            for j in range(inputs_np.shape[1]):
                ax[i, j].imshow(inputs_np[i, j, :, :])
                ax[i, j].axis('off')
                ax[i, j].set_aspect('equal')
        f.subplots_adjust(wspace=0, hspace=0)

        self.val_writer.add_figure('input', f, epoch)

        if filename is not None:
            plt.savefig(filename)
        
    def report(self, epoch, step, loss_dict):
        print('[EPOCH: {}, step: {}] | '.format(epoch, step) +
            " | ".join(key + ': {:.6f}'.format(val) for key, val in loss_dict.items()))

    def accumulate_loss(self, acc_dict, loss_dict, n_data):
        # accumulate the loss
        for key, val in loss_dict.items():
            if key not in acc_dict:
                acc_dict[key] = val / n_data
            else:
                acc_dict[key] += val / n_data

        return acc_dict

    def write_losses(self, loss_dict, epoch, tag):
        for key, val in loss_dict.items():
            if tag == "train":
                self.train_writer.add_scalar(key, val, epoch)
            elif tag == "val":
                self.val_writer.add_scalar(key, val, epoch)
