import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------
#   Basic Conv + BN + ReLU
# ---------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# ---------------------------
#       RFB Block
# ---------------------------
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 5, padding=2),
            BasicConv2d(out_channel, out_channel, 5, padding=2)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 7, padding=3),
            BasicConv2d(out_channel, out_channel, 7, padding=3)
        )

        self.conv_cat = BasicConv2d(out_channel * 4, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)
        x_res = self.conv_res(x)

        return F.relu(x_cat + x_res, inplace=True)


# ---------------------------
#  Neighbor Connection Decoder
# ---------------------------
class NCD(nn.Module):
    def __init__(self, channel):
        super(NCD, self).__init__()
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(channel * 3, channel, 3, padding=1)

        self.conv4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        x3_1 = self.conv_upsample1(x3)
        x3_1 = F.interpolate(x3_1, size=x2.size()[2:], mode='bilinear')

        x2_1 = self.conv_upsample2(x2)
        x2_1 = F.interpolate(x2_1, size=x1.size()[2:], mode='bilinear')

        x_cat2 = torch.cat([x2_1, x1], dim=1)
        x_cat2 = self.conv_concat2(x_cat2)

        x_cat3 = torch.cat([x3_1, x2, x1], dim=1)
        x_cat3 = self.conv_concat3(x_cat3)

        x_out = self.conv4(x_cat3)
        x_out = self.conv5(x_out)
        return x_out


# ---------------------------
#         SINet-V2 MAIN
# ---------------------------
class SINetV2(nn.Module):
    def __init__(self, backbone='res2net50'):
        super(SINetV2, self).__init__()

        # Res2Net backbone (from authors)
        resnet = models.resnet50(pretrained=False)

        # Use layers from resnet50 (same as authors)
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # RFB blocks
        self.rfb2 = RFB_modified(512, 32)     # layer2 output channels: 512
        self.rfb3 = RFB_modified(1024, 32)    # layer3 output channels: 1024

        # Decoder
        self.decoder = NCD(32)

    def forward(self, x):
        size = x.size()[2:]

        x1 = self.conv1(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        rfb2_out = self.rfb2(x3)
        rfb3_out = self.rfb3(x4)

        pred = self.decoder(x2, rfb2_out, rfb3_out)
        pred = F.interpolate(pred, size=size, mode='bilinear', align_corners=True)

        return pred
