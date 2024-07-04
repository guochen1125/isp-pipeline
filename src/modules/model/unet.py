import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
        has_proj=False,
        has_bn=False,
    ):
        super().__init__()


class unet_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.prelu = nn.PReLU()

        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        if x.shape[1] != 4:
            x = self.prelu(x)

        x = self.conv0(x)
        x = self.prelu(x)
        x = self.conv1(x)

        return x


class unet_res_conv_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.iden_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.prelu = nn.PReLU()

        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

    def forward(self, x):

        identity = self.iden_conv(x)
        if x.shape[1] != 4:
            x = self.prelu(x)

        x = self.conv0(x)
        x = self.prelu(x)
        x = self.conv1(x)
        x = x + identity

        return x


class unet_res_group(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv = unet_res_conv_block(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class unet_group(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv = unet_conv_block(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        output_channel = 32 * 4
        self.g1 = unet_res_conv_block(4, output_channel)
        self.g2 = unet_res_group(output_channel, 256)
        self.g3 = unet_res_group(256, 256)
        self.g4 = unet_res_group(256, 256)
        self.g5 = unet_group(256, 256)

        output_channel = 256
        self.up4_up_deconv = nn.ConvTranspose2d(
            256, output_channel, kernel_size=2, stride=2, padding=0
        )
        self.up4_conv = unet_res_conv_block(output_channel, output_channel)

        self.up3_up_deconv = nn.ConvTranspose2d(
            output_channel, output_channel, kernel_size=2, stride=2, padding=0
        )
        self.up3_conv = unet_res_conv_block(output_channel, output_channel)

        self.up2_up_deconv = nn.ConvTranspose2d(
            output_channel, output_channel, kernel_size=2, stride=2, padding=0
        )
        self.up2_conv = unet_res_conv_block(output_channel, output_channel)

        self.up1_up_deconv = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2, padding=0
        )
        self.up1_conv = unet_res_conv_block(128, 128)

        self.prelu = nn.PReLU()
        self.last_conv = nn.Conv2d(128, 4, 3, 1, 1, bias=False)

    def forward(self, inp):
        conv1 = self.g1(inp)
        conv2 = self.g2(conv1)
        conv3 = self.g3(conv2)
        conv4 = self.g4(conv3)
        conv5 = self.g5(conv4)

        up4 = self.up4_up_deconv(conv5) + conv4
        up4 = self.up4_conv(up4)

        up3 = self.up3_up_deconv(up4) + conv3
        up3 = self.up3_conv(up3)

        up2 = self.up2_up_deconv(up3) + conv2
        up2 = self.up2_conv(up2)

        up1 = self.up1_up_deconv(up2) + conv1
        up1 = self.up1_conv(up1)

        pred = self.prelu(up1)
        pred = self.last_conv(pred)

        pred = pred + inp

        return pred


def get_loss_l1(pred: torch.Tensor, label: torch.Tensor):
    B = pred.shape[0]
    L1 = F.abs(pred - label).reshape(B, -1).mean(axis=1)
    return L1.mean()
