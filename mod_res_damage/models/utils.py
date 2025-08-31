import torch.nn as nn
    
    
class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class DoubleConv2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv2dBlock(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            Conv2dBlock(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class Conv3dBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
      

class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, mid_channels=None, embed=False):
        super().__init__()
        self.embed = embed
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv3dBlock(in_channels, mid_channels, kernel_size, stride, padding),
            Conv3dBlock(mid_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        x = self.double_conv(x)