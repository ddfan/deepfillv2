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


def visualize_l1loss(model, config, writer, device, dataset, filename=None, epoch=0):
    model.eval()
    image, mask, gt, _ = zip(*[dataset[i] for i in range(min(len(dataset), config.num_vis_imgs))])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)

    with torch.no_grad():
        output = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))

    # unflatten images
    image = torch.reshape(image, (-1, config.in_channels, config.img_size, config.img_size))
    mask = torch.reshape(mask, (-1, 1, config.img_size, config.img_size))
    output = torch.reshape(output, (-1, config.out_channels, config.img_size, config.img_size))
    gt = torch.reshape(gt, (-1, 1, config.img_size, config.img_size))

    idx = config.input_map_layers.index("obstacle_occupancy")
    mae = torch.abs((gt - output) * mask)
    grid = make_grid(torch.cat([image[:, idx:idx + 1, :, :], output, gt, mae], dim=0), scale_each=True)

    writer.add_image('images', grid, epoch)

    if filename is not None:
        save_image(grid, filename)


def visualize_cvar(model, config, writer, device, dataset, filename=None, epoch=0):
    model.eval()
    inputs, mask, gt, alpha = zip(*[dataset[i, True] for i in range(min(len(dataset), config.num_vis_imgs))])
    inputs = torch.stack(inputs)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    alpha = torch.stack(alpha)

    outputs = []
    with torch.no_grad():
        for alpha_val in config.alpha_test_val:
            alpha_test = torch.ones_like(alpha) * alpha_val
            output = model(inputs.to(device), mask.to(device), alpha_test.to(device))
            output = output.to(torch.device('cpu'))
            output = torch.reshape(output, (-1, config.out_channels, config.img_size, config.img_size))
            outputs.append(output)

        # also query the given alpha from the dataset
        output = model(inputs.to(device), mask.to(device), alpha.to(device))
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
    n_samples = torch.zeros(len(config.alpha_test_val)).to(device)
    n_gt_less_than_var = torch.zeros(len(config.alpha_test_val)).to(device)
        
    for i in range(len(config.alpha_test_val)):
        if config.use_cvar_less_var:
            var = outputs[i][:, 1:2, :, :]
            cvar = outputs[i][:, 0:1, :, :] + outputs[i][:, 1:2, :, :]
        else:
            var = outputs[i][:, 0:1, :, :]
            cvar = outputs[i][:, 1:2, :, :]
    
        n_samples[i] += torch.sum(mask)
        n_gt_less_than_var[i] += torch.sum(mask * torch.lt(gt, var)) 

        var = var * mask
        cvar = cvar * mask
        vars.append(var)
        cvars.append(cvar)
    vars = torch.cat(vars, dim=1)
    cvars = torch.cat(cvars, dim=1)
    
    # print(n_gt_less_than_var / n_samples)

    # assemble varying cvar
    # alpha = torch.reshape(alpha, (-1, 1, config.img_size, config.img_size))
    if config.use_cvar_less_var:
        varying_cvar = outputs[-1][:, 0:1, :, :] + outputs[-1][:, 1:2, :, :]
    else:
        varying_cvar = outputs[-1][:, 1:2, :, :]
    img_arr = torch.cat([gt, vars, cvars, varying_cvar * mask], dim=1)

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

    writer.add_figure('output', f, epoch)

    # create input figure
    inputs_img = torch.cat((inputs, mask), dim=1)
    inputs_np = inputs_img.cpu().detach().numpy()
    f, ax = plt.subplots(inputs_np.shape[0], inputs_np.shape[1],
        figsize=(inputs_np.shape[1] * 2, inputs_np.shape[0] * 2))
    f.tight_layout()
    for i in range(inputs_np.shape[0]):
        for j in range(inputs_np.shape[1]):
            ax[i, j].imshow(inputs_np[i, j, :, :])
            ax[i, j].axis('off')
            ax[i, j].set_aspect('equal')
    f.subplots_adjust(wspace=0, hspace=0)

    writer.add_figure('input', f, epoch)

    if filename is not None:
        plt.savefig(filename)
