import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # 对输入张量inputs中的每个元素应用sigmoid激活函数,将输入的实数值压缩到(0, 1)的范围内
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        # 将inputs张量展平为一维张量
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        # 计算模型输出inputs和真实标签targets之间的二分类交叉熵损失，并且采用平均值作为最终的损失值
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = (0.5 * BCE) + (0.5 * dice_loss)

        return Dice_BCE



