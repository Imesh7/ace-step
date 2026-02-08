import torch.nn as nn

from model.autoencoder.autoencoder import conv_block


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        self.layers.append(
            conv_block(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        skip_connection = []
        for layer in self.layers:
            x = layer(x)
            skip_connection.append(x)
        return x, skip_connection
