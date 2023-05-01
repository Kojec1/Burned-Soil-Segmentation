from torch import nn, Tensor
import torch.nn.functional as F


class BCEDice(nn.Module):
    """Combination of binary cross-entropy and dice coefficient as the loss function"""

    def __init__(self) -> None:
        super(BCEDice, self).__init__()

    def forward(self, inputs: Tensor, targets: Tensor, smooth: float = 1e-5) -> float:
        # Flatten predictions and targets
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Count the intersection between predictions and targets
        intersection = (inputs * targets).sum()

        # Count the dice coefficient and binary cross entropy
        dice = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return BCE + dice


def Dice(inputs: Tensor, targets: Tensor, smooth: float = 1e-5) -> float:
    """The Dice coefficient"""

    # Flatten predictions and targets
    inputs = F.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # Count the intersection between predictions and targets
    intersection = (inputs * targets).sum()

    return 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)


def IoU(inputs: Tensor, targets: Tensor, smooth: float = 1e-5) -> float:
    """The Intersection over Union metric"""

    # Flatten predictions and targets
    inputs = F.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # Count the intersection and union between predictions and targets
    intersection = (inputs * targets).sum()
    union = (inputs + targets).sum() - intersection

    return (intersection + smooth) / (union + smooth)
