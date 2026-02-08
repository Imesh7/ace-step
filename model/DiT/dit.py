import torch.nn as nn
import numpy as np
from model.transformer.attention import CrossAttention, FeedForward, MultiHeadAttention

# Implement DiT (Diffusion Transformer)


class DiffusionTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Re-check the in_channels & out , because in multi atten we did divide the value by heads
        self.self_atten = MultiHeadAttention(
            in_channels=in_channels, out_channels=512, num_heads=8
        )
        self.cross_atten = CrossAttention(
            in_channels=in_channels, out_channels=512, num_heads=8
        )
        self.norm = nn.LayerNorm(512)
        self.mix_ff = FeedForward(in_channels=512, out_channels=512)

    def sinusoidal_positional_encoding(self, max_length, dim):
        pos = np.arange(max_length)[:, np.newaxis]

        div = np.exp(np.arange(0, dim, 2) * -np.log(10000) * dim)
        print(div)

        emb = np.zeros([max_length, dim])
        emb[:, 0::2] = np.sin(pos / div)
        emb[:, 1::2] = np.cos(pos / div)
        return emb

    def forward(self, x):
        # first flatten the latent
        x = x.flatten(start_dim=2)

        pos = self.sinusoidal_positional_encoding(x.shape[1], x.shape[2])
        x_pos = x + pos
        x_atten = self.atten(x_pos)
        x_ff = self.mix_ff(x_atten)
        return x_ff
