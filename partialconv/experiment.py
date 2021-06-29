#!/usr/bin/env python3
import os
import argparse
import torch
from torchvision import transforms

from src.dataset import CostmapDataset
from src.model import PConvUNet
from src.loss import InpaintingLoss, CvarLoss
from src.train import Trainer
from src.utils import Config, load_ckpt, create_ckpt_dir
from src.test import Tester

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    # set the config
    config = Config(args.config)

    config.data_root = args.data_root
    config.ckpt_dir_root = args.ckpt_dir_root
    
    config.input_map_layers = ["elevation",
                                "num_points",
                                "obstacle_occupancy",
                                "num_points_binned_0",
                                "num_points_binned_1",
                                "num_points_binned_2",
                                "num_points_binned_3",
                                "num_points_binned_4",
                                "robot_distance"]
    # config.output_layer = "risk_ground_truth"
    config.output_layer = "risk"
    config.variance_layer = "risk_variance"

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

    # Set the configuration for training
    if config.mode == "train":
        # Define the CostmapDataset Dataset and Data Loader
        # print("Loading the Training Dataset...")
        dataset_train = CostmapDataset(config)
        # dataset_train.clean_data()
        if len(dataset_train) == 0:
            print("No training data found!")
            quit()
        print("Training on " + str(len(dataset_train)) + " datapoints.")

        # print("Loading the Validation Dataset...")
        dataset_val = CostmapDataset(config, validation=True)
        if len(dataset_val) == 0:
            print("No validation data found!")
            quit()
        print("Validating on " + str(len(dataset_val)) + " datapoints.")
        # dataset_val.clean_data()

        # Define the Loss fucntion
        # criterion = InpaintingLoss(VGG16FeatureExtractor(),
        #                            tv_loss=config.tv_loss).to(device)
        if config.use_cvar_loss:
            criterion = CvarLoss(config).to(device)
        else:
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
            start_epoch = load_ckpt(config.resume_dir,
                                   [("model", model)],
                                   [("optimizer", optimizer)])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            print("Starting from epoch ", start_epoch)

        trainer = Trainer(start_epoch, config, device, model, dataset_train,
                          dataset_val, criterion, optimizer)
        trainer.iterate()

    # Set the configuration for testing
    elif config.mode == "test":
        dataset_test = CostmapDataset(config, test=True)
        if len(dataset_test) == 0:
            print("No test data found!")
            quit()
        print("Testing on " + str(len(dataset_test)) + " datapoints.")
        # dataset_test.clean_data()

        if config.test_model_based:
            dataset_test_variance = CostmapDataset(config, test=True, return_variance=True)
        else:
            dataset_test_variance = None
        start_epoch = load_ckpt(config.resume_dir,
                                   [("model", model)])
        tester = Tester(config, device, model, dataset_test, dataset_test_variance)
        tester.iterate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the inputs")
    parser.add_argument('--data_root', type=str, default="data")
    parser.add_argument('--ckpt_dir_root', type=str, default="data/training_logs")
    parser.add_argument('--config', type=str, default="default_config.yml")
    args = parser.parse_args()

    main(args)
