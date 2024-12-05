import torch
import torch.nn as nn
import math
import torch.nn.functional as F




class EfficientConv(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(EfficientConv, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size,
                      stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        kernel_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, kernel_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two-dimensional to one-dimensional after pooling
        y = y.squeeze(-1).transpose(-1, -2)

        # Convolutional operation
        y = self.conv(y)

        # One-dimensional to two-dimensional before sigmoid
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Perform sigmoid operation
        y = self.sigmoid(y)

        # Scale the input features based on the generated channel attention weights
        return x * y.expand_as(x)


class EffResAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, reduction=16):
        super(EffResAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        self.eca = ECA(out_channels)
        self.ghost = EfficientConv(out_channels, out_channels, kernel_size=3, ratio=2, dw_size=3, stride=1, relu=True)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.eca(x)
        x = self.ghost(x)
        if self.residual is not None:
            identity = self.residual(identity)
        x += identity
        return x




class DualPathMaskedFusion(nn.Module):
    def __init__(self, in_c, out_c):
        super(DualPathMaskedFusion, self).__init__()

        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, m):
        # 生成特征掩码
        fmask = self.fmask(x)  # 直接使用 Sigmoid 的输出作为权重

        # 对m进行最大池化，并调整尺寸以匹配x
        m = F.adaptive_max_pool2d(m, output_size=(x.shape[2], x.shape[3]))

        # 计算混合特征
        # 使用 Sigmoid 权重进行加权融合
        x1 = x * (fmask + m.clamp(min=0.5))  # 使用加法和阈值来模拟逻辑或运算

        # 经过第一条卷积路径
        x1 = self.conv1(x1)

        # 经过第二条卷积路径
        x2 = self.conv2(x)

        # 合并两个路径的结果
        x = torch.cat([x1, x2], dim=1)

        return x




class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_out = self.sigmoid_channel(avg_out + max_out)
        x = channel_out * x

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid_spatial(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = spatial_out * x

        return x


class GatedFeatureFusion(nn.Module):
    def __init__(self, in_c, out_c):
        super(GatedFeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.cbam = CBAM(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.gate = nn.Sequential(
            nn.Conv2d(out_c * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.cbam(x3)

        # Concatenate the two paths and pass through the gating mechanism
        x4 = torch.cat([x2, x3], dim=1)
        gate = self.gate(x4)
        x4 = x2 * gate + x3 * (1 - gate)

        x4 = self.relu(x4)
        return x4
