import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image


def evaluate(model, dataset, device, filename, config, experiment=None):
    print('Start the evaluation...')
    model.eval()
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))

    # unflatten images
    image = torch.reshape(image, (-1, config.in_channels, config.img_size, config.img_size))
    mask = torch.reshape(mask, (-1, 1, config.img_size, config.img_size))
    output = torch.reshape(output, (-1, config.out_channels, config.img_size, config.img_size))
    gt = torch.reshape(gt, (-1, config.out_channels, config.img_size, config.img_size))

    # output_comp = mask * image + (1 - mask) * output
    # grid = make_grid(torch.cat([image, mask, output, output_comp, gt], dim=0))
    grid = make_grid(torch.cat([image[:, 0:1, :, :], mask, output, gt], dim=0))
    save_image(grid, filename)
    if experiment is not None:
        experiment.log_image(filename, filename)
