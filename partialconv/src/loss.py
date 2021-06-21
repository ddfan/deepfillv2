import torch
import torch.nn as nn
from torchvision import models


class CvarLoss(nn.Module):
    def __init__(self, config):
        super(CvarLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.img_size = config.img_size
        self.loss_huber = config.loss_huber
        self.use_cvar_less_var = config.use_cvar_less_var

    def forward(self, input, mask, output, gt, alpha=None):
        mask = torch.reshape(mask, (-1, 1, self.img_size, self.img_size))
        gt = torch.reshape(gt, (-1, 1, self.img_size, self.img_size))
        alpha = torch.reshape(alpha, (-1, 1, self.img_size, self.img_size))
        var_and_cvar = torch.reshape(output, (-1, 2, self.img_size, self.img_size))

        var = var_and_cvar[:,0,:,:]
        if self.use_cvar_less_var:
            cvar_less_var = var_and_cvar[:, 1, :, :]
            cvar = cvar_less_var + var.detach()
        else:
            cvar = var_and_cvar[:, 1, :, :]

        var_loss = self.var_huber_loss(gt, var, alpha)
        var_error = self.var_error(gt, var, alpha)
        cvar_loss = self.l1(mask * cvar, mask * var_error.detach())

        return {'var': var_loss,
                'cvar': cvar_loss}

    def var_error(self, gt, var, alpha):
        return torch.clamp(gt - var, min=0.0) / (1.0 - alpha) + var

    def var_huber_loss(self, gt, var, alpha):
        # compute quantile loss (with custom huber smoothing)
        huber_greater = gt + 0.5 / self.loss_huber * torch.square(gt - var) + 0.5 * self.loss_huber
        huber_less = gt + 0.5 / self.loss_huber * torch.square((gt - var) * alpha / (1.0 - alpha)) + 0.5 * self.loss_huber
        no_huber = var + torch.clamp(gt - var, min=0.0) / (1.0 - alpha)

        var_geq_cost = torch.ge(var, gt)
        var_leq_cost = torch.lt(var, gt)

        huber_greater_comp = torch.lt(var, gt + self.loss_huber)
        huber_less_comp = torch.gt(var, gt - (1.0 - alpha) / alpha * self.loss_huber)

        case_greater = torch.logical_and(var_geq_cost, huber_greater_comp)
        case_less = torch.logical_and(var_leq_cost, huber_less_comp)
        case_else = torch.logical_not(torch.logical_or(case_greater, case_less))

        result = case_greater.float() * huber_greater
        result += case_less.float() * huber_less
        result += case_else.float() * no_huber

        result = torch.mean(result)
        return result

class InpaintingLoss(nn.Module):
    def __init__(self, extractor=None, tv_loss=None):
        super(InpaintingLoss, self).__init__()
        self.tv_loss = tv_loss
        self.l1 = nn.L1Loss()
        # default extractor is VGG16
        self.extractor = extractor

    def forward(self, input, mask, output, gt, alpha=None):
        # Non-hole pixels directly set to ground truth
        if self.tv_loss is not None or self.extractor is not None:
            comp = mask * input + (1 - mask) * output

        # Total Variation Regularization
        if self.tv_loss is not None:
            tv_loss = total_variation_loss(comp, mask, self.tv_loss)
            # tv_loss = (torch.mean(torch.abs(comp[:, :, :, :-1] - comp[:, :, :, 1:])) \
            #           + torch.mean(torch.abs(comp[:, :, :, 1:] - comp[:, :, :, :-1])) \
            #           + torch.mean(torch.abs(comp[:, :, :-1, :] - comp[:, :, 1:, :])) \
            #           + torch.mean(torch.abs(comp[:, :, 1:, :] - comp[:, :, :-1, :]))) / 2
        else:
            tv_loss = torch.tensor(0.0)

        # Hole Pixel Loss
        hole_loss = self.l1((1 - mask) * output, (1 - mask) * gt)

        # Valid Pixel Loss
        valid_loss = self.l1(mask * output, mask * gt)

        # Perceptual Loss and Style Loss
        if self.extractor is not None:
            feats_out = self.extractor(output)
            feats_comp = self.extractor(comp)
            feats_gt = self.extractor(gt)
            perc_loss = 0.0
            style_loss = 0.0
            # Calculate the L1Loss for each feature map
            for i in range(3):
                perc_loss += self.l1(feats_out[i], feats_gt[i])
                perc_loss += self.l1(feats_comp[i], feats_gt[i])
                style_loss += self.l1(gram_matrix(feats_out[i]),
                                      gram_matrix(feats_gt[i]))
                style_loss += self.l1(gram_matrix(feats_comp[i]),
                                      gram_matrix(feats_gt[i]))
        else:
            perc_loss = torch.tensor(0.0)
            style_loss = torch.tensor(0.0)

        return {'valid': valid_loss,
                'hole': hole_loss,
                'perc': perc_loss,
                'style': style_loss,
                'tv': tv_loss}


# The network of extracting the feature for perceptual and style loss
class VGG16FeatureExtractor(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        normalization = Normalization(self.MEAN, self.STD)
        # Define the each feature exractor
        self.enc_1 = nn.Sequential(normalization, *vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, input):
        feature_maps = [input]
        for i in range(3):
            feature_map = getattr(
                self, 'enc_{}'.format(i + 1))(feature_maps[-1])
            feature_maps.append(feature_map)
        return feature_maps[1:]


# Normalization Layer for VGG
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, input):
        # normalize img
        if self.mean.type() != input.type():
            self.mean = self.mean.to(input)
            self.std = self.std.to(input)
        return (input - self.mean) / self.std


# Calcurate the Gram Matrix of feature maps
def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def dialation_holes(hole_mask):
    b, ch, h, w = hole_mask.shape
    dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(hole_mask)
    torch.nn.init.constant_(dilation_conv.weight, 1.0)
    with torch.no_grad():
        output_mask = dilation_conv(hole_mask)
    updated_holes = output_mask != 0
    return updated_holes.float()


def total_variation_loss(image, mask, method):
    hole_mask = 1 - mask
    dilated_holes = dialation_holes(hole_mask)
    colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
    rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]
    if method == 'sum':
        loss = torch.sum(torch.abs(colomns_in_Pset * (
                    image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
            torch.sum(torch.abs(rows_in_Pset * (
                    image[:, :, :1, :] - image[:, :, -1:, :])))
    else:
        loss = torch.mean(torch.abs(colomns_in_Pset * (
                    image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
            torch.mean(torch.abs(rows_in_Pset * (
                    image[:, :, :1, :] - image[:, :, -1:, :])))
    return loss


if __name__ == '__main__':
    from config import get_config
    config = get_config()
    vgg = VGG16FeatureExtractor()
    criterion = InpaintingLoss(config['loss_coef'], vgg)

    img = torch.randn(1, 3, 500, 500)
    mask = torch.ones((1, 1, 500, 500))
    mask[:, :, 250:, :][:, :, :, 250:] = 0
    input = img * mask
    out = torch.randn(1, 3, 500, 500)
    loss = criterion(input, mask, out, img)
