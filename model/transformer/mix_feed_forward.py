import torch
import torch.nn as nn


class MixFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels)
        self.layer2 = nn.Conv1d(in_channels=in_channels, out_channels=3 * out_channels)
        self.act = nn.SiLU()
        self.layer3 = nn.Conv1d(in_channels=3 * in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        input = x

        x = self.act(x)
        x = x + input
        return self.layer3(x)
