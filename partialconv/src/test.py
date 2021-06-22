import torch
from torch.utils.data import DataLoader

from .vis import *
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter


class Tester(object):
    def __init__(self, config, device, model, dataset_test, criterion):
        self.epoch = epoch
        self.config = config
        self.device = device
        self.model = model
        self.dataset_test = dataset_test
        self.dataloader = DataLoader(dataset_test,
                                       batch_size=config.batch_size,
                                       shuffle=True)
        self.steps = len(self.dataloader)
        self.criterion = criterion

        self.test_writer = SummaryWriter(self.config.ckpt + "/test", flush_secs=1)

    def iterate(self):
        print('Start the testing...')
        first_iteration = True
        if self.config.show_progress_bar:
            dataloader = tqdm(self.dataloader_train, file=sys.stdout)
        else:
            dataloader = self.dataloader_train

        # save visualization
        if self.config.use_cvar_loss:
            visualize_cvar(self.model, self.config, self.writer, self.device, self.dataset_test, epoch=0)
        else:
            visualize_l1loss(self.model, self.config, self.writer, self.device, self.dataset_test, epoch=0)

        # compute statistics
        self.compute_statistics_cvar()

    def compute_statistics_cvar(self):
        print("ok")
