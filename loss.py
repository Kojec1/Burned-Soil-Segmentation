import torch
import torch.nn.functional as F
from torch import nn, Tensor


class BCEDiceLoss(nn.Module):
    """Combination of binary cross-entropy and dice coefficient as the loss function"""

    def __init__(self) -> None:
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs: Tensor, targets: Tensor, smooth: float = 1e-5) -> float:
        # Flatten predictions and targets
        BCE = F.binary_cross_entropy_with_logits(inputs, targets.to(torch.float32), reduction='mean')

        n = targets.size(0)

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(n, -1)
        targets = targets.view(n, -1).float()

        # Count the intersection between predictions and targets
        intersection = (inputs * targets).sum(dim=1)

        # Count the dice coefficient and binary cross entropy
        dice = (2. * intersection + smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + smooth)
        dice = 1 - dice.sum() / n

        return BCE + dice


def dice_coef(inputs: Tensor, targets: Tensor, smooth: float = 1e-5) -> float:
    """The Dice coefficient"""
    n = targets.size(0)

    # Flatten predictions and targets
    inputs = torch.sigmoid(inputs).round()
    inputs = inputs.view(n, -1)
    targets = targets.view(n, -1).float()

    # Count the intersection between predictions and targets
    intersection = (inputs * targets).sum(dim=1)

    dice = (2. * intersection + smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + smooth)

    return 1 - dice.sum() / n


def iou_score(inputs: Tensor, targets: Tensor, smooth: float = 1e-5) -> float:
    """The Intersection over Union metric"""
    n = targets.size(0)

    # Flatten predictions and targets
    inputs = torch.sigmoid(inputs).round()
    inputs = inputs.view(n, -1)
    targets = targets.view(n, -1).float()

    # Count the intersection and union between predictions and targets
    intersection = (inputs * targets).sum(dim=1)
    union = (inputs + targets).sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.sum() / n


def accuracy_score(inputs: Tensor, targets: Tensor, smooth: float = 1e-5) -> float:
    """Pixel-wise accuracy metric"""
    n = targets.size(0)

    # Flatten predictions and targets
    inputs = torch.sigmoid(inputs).round()
    inputs = inputs.view(n, -1)
    targets = targets.view(n, -1).float()

    # Count TP, FP, FN, TN
    tp = (inputs * targets).sum(dim=1)
    fp = (inputs * (1 - targets)).sum(dim=1)
    fn = ((1 - inputs) * targets).sum(dim=1)
    tn = ((1 - inputs) * (1 - targets)).sum(dim=1)

    acc = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)

    return acc.sum() / n
