import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

from mod_res_damage.models.utils import DoubleConv3d


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


class Down3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), mid_channels=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2), padding=0),
            DoubleConv3d(in_channels, out_channels, kernel_size, stride, padding, mid_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up3d(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv3d(in_channels + in_channels // 2, out_channels, in_channels // 2, stride=stride, padding=padding)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1,2,2), stride=(1,2,2))
            self.conv = DoubleConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    
class OutConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 1, 1), stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)
      

class UNet3D(nn.Module):
    _default_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",
        "moped_enable": True,
        "moped_delta": 0.5,
    }
    
    def __init__(self,
                 n_channels,
                 n_predictands,
                 num_frames=2,
                 bilinear=False,
                 kernel_size=(2,3,3),
                 embed=False,
                 stride=1,
                 padding=(1,1,1)):
        super().__init__()

        self.num_predictands = n_predictands
        self.n_channels = n_channels
        self.n_predictands = n_predictands
        self.num_frames = num_frames
        self.bilinear = bilinear
        self.embed = embed

        self.inc = DoubleConv3d(n_channels, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.down1 = Down3d(32, 64)
        self.down2 = Down3d(64, 128)
        self.down3 = Down3d(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down3d(256, 512 // factor)
        
        # self.bn = DoubleConv3d(512, 512, kernel_size=(1,3,3), stride=stride, padding=(0,1,1))
        
        self.up1 = Up3d(512, 256 // factor, bilinear=bilinear)
        self.up2 = Up3d(256, 128 // factor, bilinear=bilinear)
        self.up3 = Up3d(128, 64 // factor, bilinear=bilinear)
        self.up4 = Up3d(64, 32, bilinear=bilinear)
        self.outc = OutConv3d(32, n_predictands, kernel_size=(num_frames, 1, 1))
        dnn_to_bnn(self, self._default_prior_parameters)

    def forward(self, x):
        x = x['S1GRD']
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # x6 = self.bn(x5)
        if self.embed:
            embed = x5.flatten(2).transpose(1,2)
            return embed

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x).squeeze(2)
        return logits