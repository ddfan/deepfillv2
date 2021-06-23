import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        # Inherit the parent class (Conv2d)
        super(PartialConv2d, self).__init__(in_channels, out_channels,
                                            kernel_size, stride=stride,
                                            padding=padding, dilation=dilation,
                                            groups=groups, bias=bias,
                                            padding_mode=padding_mode)
        # Define the kernel for updating mask
        self.mask_kernel = torch.ones(self.out_channels, self.in_channels,
                                      self.kernel_size[0], self.kernel_size[1])
        # Define sum1 for renormalization
        self.sum1 = self.mask_kernel.shape[1] * self.mask_kernel.shape[2] \
                                              * self.mask_kernel.shape[3]
        # Define the updated mask
        self.update_mask = None
        # Define the mask ratio (sum(1) / sum(M))
        self.mask_ratio = None
        # Initialize the weights for image convolution
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, img, mask):
        with torch.no_grad():
            if self.mask_kernel.type() != img.type():
                self.mask_kernel = self.mask_kernel.to(img)
            # Create the updated mask
            # for calcurating mask ratio (sum(1) / sum(M))
            self.update_mask = F.conv2d(mask, self.mask_kernel,
                                        bias=None, stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        groups=1)
            # calcurate mask ratio (sum(1) / sum(M))
            self.mask_ratio = self.sum1 / (self.update_mask + 1e-8)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # calcurate WT . (X * M)
        conved = torch.mul(img, mask)
        conved = F.conv2d(conved, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        if self.bias is not None:
            # Maltuply WT . (X * M) and sum(1) / sum(M) and Add the bias
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(conved - bias_view, self.mask_ratio) + bias_view
            # The masked part pixel is updated to 0
            output = torch.mul(output, self.mask_ratio)
        else:
            # Multiply WT . (X * M) and sum(1) / sum(M)
            output = torch.mul(conved, self.mask_ratio)

        return output, self.update_mask


class UpsampleConcat(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the upsampling layer with nearest neighbor
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, dec_feature, enc_feature, dec_mask, enc_mask):
        # upsample and concat features
        out = self.upsample(dec_feature)
        out = torch.cat([out, enc_feature], dim=1)
        # upsample and concat masks
        out_mask = self.upsample(dec_mask)
        out_mask = torch.cat([out_mask, enc_mask], dim=1)
        return out, out_mask


class PConvActiv(nn.Module):
    def __init__(self, in_ch, out_ch, sample='none-3', dec=False,
                 bn=True, active='relu', conv_bias=False):
        super().__init__()
        # Define the partial conv layer
        if sample == 'down-7':
            params = {"kernel_size": 7, "stride": 2, "padding": 3}
        elif sample == 'down-5':
            params = {"kernel_size": 5, "stride": 2, "padding": 2}
        elif sample == 'down-3':
            params = {"kernel_size": 3, "stride": 2, "padding": 1}
        else:
            params = {"kernel_size": 3, "stride": 1, "padding": 1}
        self.conv = PartialConv2d(in_ch, out_ch,
                                  params["kernel_size"],
                                  params["stride"],
                                  params["padding"],
                                  bias=conv_bias)

        # Define other layers
        if dec:
            self.upcat = UpsampleConcat()
        if bn:
            bn = nn.BatchNorm2d(out_ch)
        if active == 'relu':
            self.activation = nn.ReLU()
        elif active == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif active == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, img, mask, enc_img=None, enc_mask=None):
        if hasattr(self, 'upcat'):
            out, update_mask = self.upcat(img, enc_img, mask, enc_mask)
            out, update_mask = self.conv(out, update_mask)
        else:
            out, update_mask = self.conv(img, mask)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        if hasattr(self, 'activation'):
            out = self.activation(out)
        return out, update_mask


class PConvUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.freeze_enc_bn = True if config.finetune else False
        self.layer_size = config.layer_size
        if config.use_cvar_loss:
            self.img_channels = config.in_channels
            self.in_channels = config.in_channels + 1
        else:
            self.img_channels = config.in_channels
            self.in_channels = config.in_channels
        
        self.out_channels = config.out_channels
        self.img_size = config.img_size
        self.map_layers = config.input_map_layers
        self.use_cvar_loss = config.use_cvar_loss
        self.print_sizes = False
        n_feat = 256

        self.enc_1 = PConvActiv(self.in_channels, 64, 'down-7', bn=False)
        self.enc_2 = PConvActiv(64, 128, 'down-5')
        self.enc_3 = PConvActiv(128, 256, 'down-5')
        self.enc_4 = PConvActiv(256, n_feat, 'down-3')
        self.enc_5 = PConvActiv(n_feat, n_feat, 'down-3')
        self.enc_6 = PConvActiv(n_feat, n_feat, 'down-3')
        self.enc_7 = PConvActiv(n_feat, n_feat, 'down-3')
        self.enc_8 = PConvActiv(n_feat, n_feat, 'down-3')

        self.dec_8 = PConvActiv(n_feat + n_feat, n_feat, dec=True, active='leaky')
        self.dec_7 = PConvActiv(n_feat + n_feat, n_feat, dec=True, active='leaky')
        self.dec_6 = PConvActiv(n_feat + n_feat, n_feat, dec=True, active='leaky')
        self.dec_5 = PConvActiv(n_feat + n_feat, n_feat, dec=True, active='leaky')
        self.dec_4 = PConvActiv(n_feat + 256, 256, dec=True, active='leaky')
        self.dec_3 = PConvActiv(256 + 128, 128, dec=True, active='leaky')
        self.dec_2 = PConvActiv(128 + 64, 64, dec=True, active='leaky')
        self.dec_1 = PConvActiv(64 + self.in_channels, self.out_channels, dec=True, bn=False,
                                active='leaky', conv_bias=True)

    def forward(self, img, mask, alpha=None):
        # reshape from flat to proper
        img = torch.reshape(img, (-1, self.img_channels, self.img_size, self.img_size))
        mask = torch.reshape(mask, (-1, 1, self.img_size, self.img_size))
        if alpha is not None and self.use_cvar_loss:
            alpha = torch.reshape(alpha, (-1, 1, self.img_size, self.img_size))
            # append alpha channel to img
            img = torch.cat((img, alpha), 1)

        # normalize image
        img = self.normalize_img(img, mask)

        # tile mask
        mask = mask.repeat(1,self.in_channels, 1, 1)
        
        if self.print_sizes:
            print(img.size(), mask.size())

        enc_f, enc_m = [img], [mask]
        for layer_num in range(1, self.layer_size + 1):
            if layer_num == 1:
                feature, update_mask = \
                    getattr(self, 'enc_{}'.format(layer_num))(img, mask)
            else:
                enc_f.append(feature)
                enc_m.append(update_mask)
                feature, update_mask = \
                    getattr(self, 'enc_{}'.format(layer_num))(feature,
                                                              update_mask)
            if self.print_sizes:
                print(feature.size(), update_mask.size())

        assert len(enc_f) == self.layer_size

        for layer_num in reversed(range(1, self.layer_size + 1)):
            feature, update_mask = getattr(self, 'dec_{}'.format(layer_num))(
                    feature, update_mask, enc_f.pop(), enc_m.pop())
            if self.print_sizes:
                print(feature.size(), update_mask.size())

        # flatten output
        feature = torch.reshape(feature, (-1, self.out_channels, self.img_size * self.img_size))

        return feature

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters
        In initial training, BN set to True
        In fine-tuning stage, BN set to False
        """
        super().train(mode)
        if not self.freeze_enc_bn:
            return
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                module.eval()

    def normalize_img(self, img, mask):
        elevation_copy = img[:, self.map_layers.index("elevation"), ...].clone()
        elevation_copy[elevation_copy == 0] = float("nan")
        median_elevation = elevation_copy.nanmedian(axis=-1)[0].nanmedian(axis=-1)[0].nan_to_num()

        for i, layer in enumerate(self.map_layers):
            if "num_points" in layer:
                img[:, i, ...] = 1.0 - torch.exp(-img[:, i, ...] / 5.0)
            elif "elevation" in layer:
                img[:, i, ...] = img[:, i, ...] - median_elevation.reshape((-1, 1, 1))
            elif "distance" in layer:
                img[:, i, ...] = img[:, i, ...] / 25.0

        return img

if __name__ == '__main__':
    from utils import init_xavier
    size = (1, 3, 512, 512)
    img = torch.ones(size)
    mask = torch.ones(size)
    mask[:, :, 128:-128, :][:, :, :, 128:-128] = 0

    conv = PartialConv2d(3, 3, 3, 1, 1)
    criterion = nn.L1Loss()
    img.requires_grad = True

    output, out_mask = conv(img, mask)
    loss = criterion(output, torch.randn(size))
    loss.backward()

    # print(img.grad[0])
    assert (torch.sum(torch.isnan(conv.weight.grad)).item() == 0)
    assert (torch.sum(torch.isnan(conv.bias.grad)).item() == 0)

    model = PConvUNet()
    before = model.enc_5.conv.weight[0][0]
    print(before)
    # model.apply(init_xavier)
    # after = model.enc_5.conv.weight[0][0]
    # print(after - before)
    output, out_mask = model(img, mask)
