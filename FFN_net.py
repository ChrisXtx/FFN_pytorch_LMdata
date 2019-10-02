__all__ = ['FFN']

import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels=32, mid_channels=32, kernel_size=(3, 3, 3), padding=1):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Conv3d(in_channels, mid_channels, kernel_size, padding=padding)
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size, padding=padding)

    def forward(self, x):
        conv0_out = self.conv0(F.relu(x))
        conv1_out = self.conv1(conv0_out)

        return conv1_out + F.relu(x)


class FFN(nn.Module):
    def __init__(self, in_channels=2, mid_channels=32, out_channels=2, kernel_size=(3, 3, 3), padding=1, depth=12):
        super(FFN, self).__init__()

        self.conv0 = nn.Conv3d(in_channels, mid_channels, kernel_size, padding=padding)
        self.conv1 = nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding)
        self.resblocks = nn.Sequential(*[ResBlock(mid_channels, mid_channels, kernel_size, padding) for i in range(depth)])
        self.conv3 = nn.Conv3d(mid_channels, out_channels, (1, 1, 1))

        self._init_weights()

    def forward(self, x):
        conv0_out = self.conv0(x)
        conv1_out = self.conv1(conv0_out)
        res_out = self.resblocks(conv1_out)
        logits = self.conv3(F.relu(res_out))

        return logits

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
