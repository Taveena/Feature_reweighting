from typing import List

import torch
import torch.nn as nn

from base.layers import Conv2dWithConstraint, LinearWithConstraint
from utils.utils import initialize_weight

torch.set_printoptions(linewidth=1000)


class DeepConvNet(nn.Module):
    def __init__(
            self,
            n_classes: int,
            input_shape: List[int],
            first_conv_length: int,
            block_out_channels: List[int],
            pool_size: int,
            last_dim: int,
            weight_init_method=None,
    ) -> None:
        super(DeepConvNet, self).__init__()
        b, c, s, t = input_shape

        self.first_conv_block = nn.Sequential(
            Conv2dWithConstraint(1, block_out_channels[0], kernel_size=(1, first_conv_length), max_norm=2),
            Conv2dWithConstraint(block_out_channels[0], block_out_channels[1], kernel_size=(s, 1), bias=False,
                                 max_norm=2),
            nn.BatchNorm2d(block_out_channels[1]),
            nn.ELU(),
            nn.MaxPool2d((1, pool_size))
        )

        self.deep_block = nn.ModuleList(
            [self.default_block(block_out_channels[i - 1], block_out_channels[i], first_conv_length, pool_size) for i in
             range(2, 5)]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(last_dim, n_classes, max_norm=0.5)  # time points = 1125
        )

        #initialize_weight(self, weight_init_method)

    def default_block(self, in_channels, out_channels, T, P):
        default_block = nn.Sequential(
            nn.Dropout(0.5),
            Conv2dWithConstraint(in_channels, out_channels, (1, T), bias=False, max_norm=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d((1, P))
        )
        return default_block

    def forward(self, x):
        out = self.first_conv_block(x)
        for block in self.deep_block:
            out = block(out)
        out = self.classifier(out)
        return out

    def weights_init(self):
        def weights_init(m):
             if isinstance(m, nn.Conv2d):
                 torch.nn.init.xavier_uniform(m.weight.data)
             if isinstance(m,nn.Linear):
                 torch.nn.init.xavier_uniform(m.weight.data)
                 m.bias.data.fill_(0.01) 



if __name__=="__main__":
	batch_size = 5
	chans =22
	#net = EEGNet_fusion(4,Chans=chans, Samples=1000)
	net = DeepConvNet(n_classes=4,input_shape=[batch_size,1,chans,1000],first_conv_length=10,block_out_channels=[25, 25, 50, 100,200],pool_size=3,last_dim=1400)
	from torchinfo import summary
	summary(net.cuda(), (batch_size, 1,chans, 1000))
