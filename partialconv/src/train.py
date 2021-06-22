import torch
from torch.utils.data import DataLoader

from .vis import *
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter


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
                visualize_cvar(self.model, self.config, self.val_writer, self.device, self.dataset_val, epoch=self.epoch)
            else:
                visualize_l1loss(self.model, self.config, self.val_writer, self.device, self.dataset_val, epoch=self.epoch)

            # save the model
            new_val_loss = val_loss_dict[self.config.save_model_loss]
            if new_val_loss < best_val_loss or i + 1 == self.config.max_epochs or i % self.config.save_model_interval == 0:
                if best_val_loss != float("inf"):
                    if new_val_loss < best_val_loss:
                        print('Validation loss improved from {:.6f} to {:.6f}, saving.'.format(best_val_loss, new_val_loss))
                    else:
                        print('Saving regular checkpoint, loss is {:.6f}.'.format(new_val_loss))

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

        # numerical differentiation
        if self.config.use_monotonic_loss:
            with torch.no_grad():
                output_alpha_plus = self.model(input, mask, alpha + self.config.monotonic_loss_delta)
        else:
            output_alpha_plus = None

        loss_dict = self.criterion(input, mask, output, gt, alpha, output_alpha_plus)
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
