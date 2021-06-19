#!/usr/bin/env python3

# from comet_ml import Experiment

import argparse
import torch
from torchvision import transforms

from src.dataset import InpaintDataset
from src.model import PConvUNet
from src.loss import InpaintingLoss, VGG16FeatureExtractor
from src.train import Trainer
from src.utils import Config, load_ckpt, create_ckpt_dir

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    # set the config
    config = Config("default_config.yml")
    config.ckpt = create_ckpt_dir(config.ckpt_dir_root)
    print("Check Point is '{}'".format(config.ckpt))

    # Define the used device
    device = torch.device("cuda:{}".format(config.cuda_id)
                          if torch.cuda.is_available() else "cpu")

    # Define the model
    print("Loading the Model...")
    model = PConvUNet(config)
    if config.finetune:
        model.load_state_dict(torch.load(config.finetune)['model'])
    model.to(device)

    print("Model has {} parameters".format(count_parameters(model)))

    # # Data Transformation
    # img_tf = transforms.Compose([
    #             transforms.ToTensor()
    #             ])
    # if config.mask_augment:
    #     mask_tf = transforms.Compose([
    #                 transforms.RandomResizedCrop(256),
    #                 transforms.ToTensor()
    #                 ])
    # else:
    #     mask_tf = transforms.Compose([
    #                 transforms.ToTensor()
    #                 ])

    # # Define the Validation set
    # dataset_val = Places2(config.data_root,
    #                       img_tf,
    #                       mask_tf,
    #                       data="val")

    # print("Loading the Validation Dataset...")
    dataset_val = InpaintDataset(config, validation=True)
    print("Validating on " + str(len(dataset_val)) + " datapoints.")
    
    # Set the configuration for training
    if config.mode == "train":
        # set the comet-ml
        if config.comet:
            print("Connecting to Comet ML...")
            experiment = Experiment(api_key=config.api_key,
                                    project_name=config.project_name,
                                    workspace=config.workspace)
            experiment.log_parameters(config.__dict__)
        else:
            experiment = None

        # Define the InpaintDataset Dataset and Data Loader
        # print("Loading the Training Dataset...")
        dataset_train = InpaintDataset(config)
        print("Training on " + str(len(dataset_train)) + " datapoints.")

        # Define the Loss fucntion
        # criterion = InpaintingLoss(VGG16FeatureExtractor(),
        #                            tv_loss=config.tv_loss).to(device)
        criterion = InpaintingLoss().to(device)

        # Define the Optimizer
        lr = config.finetune_lr if config.finetune else config.initial_lr
        if config.optim == "Adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                model.parameters()),
                                         lr=lr,
                                         weight_decay=config.weight_decay)
        elif config.optim == "SGD":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                               model.parameters()),
                                        lr=lr,
                                        momentum=config.momentum,
                                        weight_decay=config.weight_decay)

        start_epoch = 0
        if config.resume:
            print("Loading the trained params and the state of optimizer...")
            start_epoch = load_ckpt(config.resume,
                                   [("model", model)],
                                   [("optimizer", optimizer)])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            print("Starting from epoch ", start_epoch)

        trainer = Trainer(start_epoch, config, device, model, dataset_train,
                          dataset_val, criterion, optimizer, experiment=experiment)
        if config.comet:
            with experiment.train():
                trainer.iterate()
        else:
            trainer.iterate()

    # Set the configuration for testing
    elif config.mode == "test":
        pass
        # <model load the trained weights>
        # evaluate(model, dataset_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the inputs")
    parser.add_argument('--train', type=bool, default=True)
    args = parser.parse_args()

    main(args)
