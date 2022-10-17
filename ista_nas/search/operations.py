import torch
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

__all__ = ["OPS", "ReLUConvBN", "DilConv", "SepConv", "Identity", "Zero", "FactorizedReduce"]


OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(
        3, stride=stride, padding=1),
    'skip_connect' : lambda C, stride, affine: (Identity() if stride == 1 
        else FactorizedReduce(C, C, affine=affine)),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv(
        C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5' : lambda C, stride, affine: SepConv(
        C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7' : lambda C, stride, affine: SepConv(
        C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv(
        C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5' : lambda C, stride, affine: DilConv(
        C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, 2*C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(2*C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
}


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x).to(device)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, 2*C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(2*C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x).to(device)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, 2*C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(2*C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(2*C_in, 2*C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(2*C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))
        torch.save(self.op,'/root/data/nop.pt')
    def forward(self, x):
        return self.op(x).to(device)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x.to(device)


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.).to(device)
        return x[:,:,::self.stride,::self.stride].mul(0.).to(device)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x).to(device)#cat按维数1拼接（横着拼）
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out=out.to(device)#torch.Size([256, 64, 32, 32])
        out = self.bn(out).to(device)#torch.Size([256, 32, 16, 16])
        return out.to(device)
