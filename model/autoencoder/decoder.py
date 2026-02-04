import torch
import torch.nn as nn

from model.autoencoder.utils import conv_block

class Decoder(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.layers = nn.ModuleList()
    self.layers.append(conv_block(in_channels=in_channels, out_channels=out_channels))

  def forward(self, x, skip_connections):
    skip_connection = skip_connections[::-1]
    for layer in self.layers:
      x = layer(x)
      x = torch.cat([x, skip_connection.pop(0)], dim=1)
    return x