import torch
import torch.nn as nn
from model.blocks_cs1 import EffResAttention, GatedFeatureFusion, DualPathMaskedFusion


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, name=None):
        super(EncoderBlock, self).__init__()
        self.name = name
        self.e1 = EffResAttention(in_c, out_c)
        self.m1 = DualPathMaskedFusion(out_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, masks):
        x = self.e1(inputs)
        p = self.m1(x, masks)
        o = self.pool(p)
        return o, x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, name=None):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1)
        self.g1 = GatedFeatureFusion(in_c+in_c, out_c)
        self.g2 = GatedFeatureFusion(out_c, out_c)
        self.m1 = DualPathMaskedFusion(out_c, out_c)

    def forward(self, inputs, skip, masks):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        # x = torch.cat([x, skip], axis=1)
        x = self.g1(x)
        x = self.g2(x)
        p = self.m1(x, masks)
        return p


class FANet(nn.Module):
    def __init__(self):
        super(FANet, self).__init__()
        self.e1 = EncoderBlock(3, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)

        self.d1 = DecoderBlock(256, 128)
        self.d2 = DecoderBlock(128, 64)
        self.d3 = DecoderBlock(64, 32)
        self.d4 = DecoderBlock(32, 16)

        self.output = nn.Conv2d(16+1, 1, kernel_size=1, padding=0)

    def forward(self, x):
        inputs, masks = x[0], x[1]

        p1, s1 = self.e1(inputs, masks)
        p2, s2 = self.e2(p1, masks)
        p3, s3 = self.e3(p2, masks)
        p4, s4 = self.e4(p3, masks)

        d1 = self.d1(p4, s4, masks)
        d2 = self.d2(d1, s3, masks)
        d3 = self.d3(d2, s2, masks)
        d4 = self.d4(d3, s1, masks)

        d5 = torch.cat([d4, masks], dim=1)
        # d5 = torch.cat([d4, masks], axis=1)
        output = self.output(d5)

        return output


if __name__ == "__main__":
    if torch.cuda.is_available():
        x = torch.randn((2, 3, 256, 256)).cuda()
        m = torch.randn((2, 1, 256, 256)).cuda()
        model = FANet().cuda()
        y = model([x, m])
    else:
        x = torch.randn((2, 3, 256, 256))  # 不再使用.cuda()将其移到CPU
        m = torch.randn((2, 1, 256, 256))  # 不再使用.cuda()将其移到CPU
        model = FANet()  # 不再使用.cuda()将其移到CPU
        y = model([x, m])
    print(y.shape)
