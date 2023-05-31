import torch
import torchvision
from torch import nn, Tensor
import torch.nn.functional as F


class NestedUNet(nn.Module):
    """Nested U-Net architecture"""

    def __init__(self, n_classes: int, output_size: tuple, deep_supervision: bool = True) -> None:
        super(NestedUNet, self).__init__()
        self.output_size = output_size

        # Backbone network
        self.backbone = Backbone()
        # Decoder network
        self.decoder = Decoder(depth=4, channels=(512, 256, 128, 64, 64))

        self.deep_supervision = deep_supervision
        # Initiate output layers
        if self.deep_supervision:
            self.out = nn.ModuleList(4 * [nn.Conv2d(64, n_classes, kernel_size=1)])
        else:
            self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # Forward pass through the backbone
        x = self.backbone(x)
        # Forward pass through the decoder network
        x = self.decoder(x)

        if self.deep_supervision:
            # Deep supervision mode
            outputs = [F.interpolate(self.out[i](x[i]), self.output_size) for i in range(4)]
            return torch.stack(outputs)

        # Single output mode
        return F.interpolate(self.out(x[-1]), self.output_size)


class Backbone(nn.Module):
    """Backbone module"""

    def __init__(self) -> None:
        super(Backbone, self).__init__()
        # Load the pre-trained ResNet34 model
        self.base_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
        # Disable learning for pre-trained model's layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get the weights of the first layer
        pretrained_weight = self.base_model.conv1.weight
        new_feature = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
        # Initialise weights for the new layer with Gaussian
        new_feature.weight.data.normal_(0, 0.001)
        # For RGB channels assign pretrained weights
        new_feature.weight.data[:, 1:4, :, :] = torch.flip(pretrained_weight, (1,))
        self.base_model.conv1 = new_feature

    def forward(self, x: Tensor) -> list:
        # Initiate a list for outputs
        outputs = list()
        # Iterate over backbone's modules and save outputs from blocks
        for module in list(self.base_model.children())[:-2]:
            x = module(x)
            if isinstance(module, (nn.ReLU, nn.Sequential)):
                outputs.append(x)

        # Return a list of outputs
        return outputs


class Decoder(nn.Module):
    """Decoder module"""

    def __init__(self, depth: int, channels: tuple) -> None:
        super(Decoder, self).__init__()
        self.depth = depth
        # Initiate a upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Initiate a list of convolutional blocks
        self.blocks = nn.ModuleList([ConvBlock(channels[i - 1] + channels[i] * (j + 1), channels[i])
                                     for i in range(1, self.depth + 1) for j in range(i)])
        # Initiate a concatenation dropout block
        self.concat_dropout = ConcatDropout()

    def forward(self, outputs: list) -> list:
        # First layer
        x3_1 = self.blocks[0](self.concat_dropout([outputs[3], self.upsample(outputs[4])]))
        # Second layer
        x2_1 = self.blocks[1](self.concat_dropout([outputs[2], self.upsample(outputs[3])]))
        x2_2 = self.blocks[2](self.concat_dropout([outputs[2], x2_1, self.upsample(x3_1)]))
        # Third layer
        x1_1 = self.blocks[3](self.concat_dropout([outputs[1], self.upsample(outputs[2])]))
        x1_2 = self.blocks[4](self.concat_dropout([outputs[1], x1_1, self.upsample(x2_1)]))
        x1_3 = self.blocks[5](self.concat_dropout([outputs[1], x1_1, x1_2, self.upsample(x2_2)]))
        # Fourth layer
        x0_1 = self.blocks[6](self.concat_dropout([outputs[0], self.upsample(outputs[1])]))
        x0_2 = self.blocks[7](self.concat_dropout([outputs[0], x0_1, self.upsample(x1_1)]))
        x0_3 = self.blocks[8](self.concat_dropout([outputs[0], x0_1, x0_2, self.upsample(x1_2)]))
        x0_4 = self.blocks[9](self.concat_dropout([outputs[0], x0_1, x0_2, x0_3, self.upsample(x1_3)]))

        # Return a list of outputs from the fourth layer
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


class ConcatDropout(nn.Module):
    """Concatenation + Dropout module"""

    def __init__(self) -> None:
        super(ConcatDropout, self).__init__()
        # Initiate a dropout layer
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: list) -> Tensor:
        # Concatenate multiple filter banks
        x = torch.concat(x, dim=1)
        # Apply dropout to the concatenated tensor
        return self.dropout(x)
