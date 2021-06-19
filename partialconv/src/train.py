import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from .utils import save_ckpt, to_items
from tqdm import tqdm
import sys

class Trainer(object):
    def __init__(self, epoch, config, device, model, dataset_train,
                 dataset_val, criterion, optimizer, experiment):
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
        self.experiment = experiment

    def iterate(self):
        print('Start the training...')
        for i in range(self.config.max_epochs):
            tqdm_dataloader = tqdm(self.dataloader_train, file=sys.stdout)
            for step, (input, mask, gt) in enumerate(tqdm_dataloader):
                loss_dict = self.evaluate(input, mask, gt) # train network
                # report the training loss
                if self.config.print_step_losses and step < self.steps - 1:
                    self.report(self.epoch, step, loss_dict)

            # determine validation loss
            val_loss_dict = {}
            for step, (input, mask, gt) in enumerate(self.dataloader_val):
                loss_dict = self.evaluate(input, mask, gt, train=False)
                # accumulate the loss
                for key, val in loss_dict.items():
                    if key not in val_loss_dict: 
                        val_loss_dict[key] = val / len(self.dataloader_val)
                    else:
                        val_loss_dict[key] += val / len(self.dataloader_val)
                self.report(self.epoch, step, val_loss_dict)

            # save visualization
            if i % self.config.vis_interval == 0:
                self.visualize(self.dataset_val,
                              '{}/val_vis/epoch_{}.png'.format(self.config.ckpt,
                                                         self.epoch))

            # save the model
            if i % self.config.save_model_interval == 0 \
                    or i + 1 == self.config.max_epochs:
                print('Saving the model...')
                save_ckpt('{}/models/{}.pth'.format(self.config.ckpt,
                                                    self.epoch),
                          [('model', self.model)],
                          [('optimizer', self.optimizer)],
                          self.epoch)

            self.epoch += 1

    def evaluate(self, input, mask, gt, train=True):
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

        # model forward
        output, _ = self.model(input, mask)
        loss_dict = self.criterion(input, mask, output, gt)
        loss = 0.0
        for key, val in loss_dict.items():
            coef = getattr(self.config, '{}_coef'.format(key))
            loss += coef * val

        if train:
            # updates the model's params
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_dict['total'] = loss
        return to_items(loss_dict)

    def visualize(self, dataset, filename):
        self.model.eval()
        image, mask, gt = zip(*[dataset[i] for i in range(min(len(dataset), self.config.num_vis_imgs))])
        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)
        with torch.no_grad():
            output, _ = self.model(image.to(self.device), mask.to(self.device))
        output = output.to(torch.device('cpu'))

        # unflatten images
        image = torch.reshape(image, (-1, self.config.in_channels, self.config.img_size, self.config.img_size))
        mask = torch.reshape(mask, (-1, 1, self.config.img_size, self.config.img_size))
        output = torch.reshape(output, (-1, self.config.out_channels, self.config.img_size, self.config.img_size))
        gt = torch.reshape(gt, (-1, self.config.out_channels, self.config.img_size, self.config.img_size))

        # output_comp = mask * image + (1 - mask) * output
        # grid = make_grid(torch.cat([image, mask, output, output_comp, gt], dim=0))
        grid = make_grid(torch.cat([image[:, 0:1, :, :], mask, output, gt], dim=0))
        save_image(grid, filename)
        if self.experiment is not None:
            self.experiment.log_image(filename, filename)

    def report(self, epoch, step, loss_dict):
        print('[EPOCH: {:>6}, STEP: {:>6}] | Valid Loss: {:.6f} | Hole Loss: {:.6f}'
              '| TV Loss: {:.6f} | Perc Loss: {:.6f}'
              '| Style Loss: {:.6f} | Total Loss: {:.6f}'.format(epoch,
                        step, loss_dict['valid'], loss_dict['hole'],
                        loss_dict['tv'], loss_dict['perc'],
                        loss_dict['style'], loss_dict['total']))
        if self.experiment is not None:
            self.experiment.log_metrics(loss_dict, step=step)
