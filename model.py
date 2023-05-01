import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.transforms import CenterCrop


class NestedUNet(nn.Module):
    def __init__(self, n_classes: int, deep_supervision: bool = True) -> None:
        super(NestedUNet, self).__init__()
        self.backbone = Backbone()
        self.decoder = Decoder(depth=4, channels=(512, 256, 128, 64, 64))

        self.deep_supervision = deep_supervision

        if self.deep_supervision:
            self.out = nn.ModuleList(4 * [nn.Conv2d(64, n_classes, kernel_size=1)])
        else:
            self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.decoder(x)

        if self.deep_supervision:
            out1 = self.out[0](x[0])
            out2 = self.out[1](x[1])
            out3 = self.out[2](x[2])
            out4 = self.out[3](x[3])

            return torch.stack([out1, out2, out3, out4])

        return self.out(x[-1])


class Backbone(nn.Module):
    def __init__(self) -> None:
        super(Backbone, self).__init__()
        self.base_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> list:
        outputs = list()
        for module in list(self.base_model.children())[:-2]:
            x = module(x)
            if isinstance(module, (nn.ReLU, nn.Sequential)):
                outputs.append(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, depth: int, channels: tuple) -> None:
        super(Decoder, self).__init__()
        self.depth = depth
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.blocks = nn.ModuleList([ConvBlock(channels[i-1] + channels[i]*(j+1), channels[i])
                                     for i in range(1, self.depth+1) for j in range(i)])

    def forward(self, outputs: list) -> list:
        x3_1 = self.blocks[0](torch.concat([outputs[3], self.upsample(outputs[4])], dim=1))

        x2_1 = self.blocks[1](torch.concat([outputs[2], self.upsample(outputs[3])], dim=1))
        x2_2 = self.blocks[2](torch.concat([outputs[2], x2_1, self.upsample(x3_1)], dim=1))

        x1_1 = self.blocks[3](torch.concat([outputs[1], self.upsample(outputs[2])], dim=1))
        x1_2 = self.blocks[4](torch.concat([outputs[1], x1_1, self.upsample(x2_1)], dim=1))
        x1_3 = self.blocks[5](torch.concat([outputs[1], x1_1, x1_2, self.upsample(x2_2)], dim=1))

        x0_1 = self.blocks[6](torch.concat([outputs[0], self.upsample(outputs[1])], dim=1))
        x0_2 = self.blocks[7](torch.concat([outputs[0], x0_1, self.upsample(x1_1)], dim=1))
        x0_3 = self.blocks[8](torch.concat([outputs[0], x0_1, x0_2, self.upsample(x1_2)], dim=1))
        x0_4 = self.blocks[9](torch.concat([outputs[0], x0_1, x0_2, x0_3, self.upsample(x1_3)], dim=1))

        return [x0_1, x0_2, x0_3, x0_4]


class ConvBlock(nn.Module):
    """ Convolutional Block module"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            # 1st convolutional layer: (in_channels, H, W) => (out_channels, H, W)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 2nd convolutional layer: (out_channels, H, W) => (out_channels, H, W)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Return the input with applied transformations
        return self.block(x)
    