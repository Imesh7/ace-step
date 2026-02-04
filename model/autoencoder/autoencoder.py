# Implement 'Deep Compression AutoEncoder'(DC-AE) to achieve highly compact mel-spectrogram latent representation

# https://arxiv.org/pdf/2410.10733

# pls refer
# 1. https://github.com/mit-han-lab/efficientvit
# 2. https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/models/efficientvit/dc_ae.py
# 3. https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/models/nn/ops.py

# You may asked why not the `SD-VAE` , I think because, when on high dimention I saw drops image quality (check out
# the above paper they figured out that issue)

import torch
import torch.nn as nn

class DeepCompressionAutoEncoder(nn.Module):
  def __init__(self,in_channels, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.encoder = Encoder(in_channels=in_channels, out_channels=32 * in_channels)
    self.decoder = Decoder(in_channels=32 * in_channels, out_channels=in_channels)

  def forward(self, x):
    x , skip_connections = self.encoder(x)
    x = self.decoder(x, skip_connections)
    return x


def conv_block(in_channels, out_channels):
  x = nn.Sequential(
    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(out_channels),
  )

  return x

class Encoder(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.layers = nn.ModuleList()
    self.layers.append(conv_block(in_channels=in_channels, out_channels=out_channels))

  def forward(self, x):
    skip_connection = []
    for layer in self.layers:
      x = layer(x)
      skip_connection.append(x)
    return x, skip_connection



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