import torch.nn as nn


def conv_block(in_channels, out_channels):
    x = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        ),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )

    return x
