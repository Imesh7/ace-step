import torch
import torch.nn as nn
import numpy as np
from model.transformer.attention import LinearAttention
from model.transformer.cross_attention import CrossAttention
from model.transformer.mix_feed_forward import MixFeedForward

# Implement DiT (Diffusion Transformer)
# This will predict the noise added to the latent, so that we can denoise it in the reverse process

class DiffusionTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Re-check the in_channels & out , because in multi atten we did divide the value by heads
        self.linear_atten = LinearAttention(
            in_channels=in_channels, out_channels=512, num_heads=8
        )
        self.cross_atten = CrossAttention(
            in_channels=in_channels, out_channels=512, num_heads=8, rope_enabled=True
        )
        self.norm = nn.LayerNorm(512)
        self.mix_ff = MixFeedForward(in_channels=512, out_channels=512)

    def sinusoidal_positional_encoding(self, max_length, dim):
        pos = np.arange(max_length)[:, np.newaxis]

        div = np.exp(np.arange(0, dim, 2) * -np.log(10000) * dim)
        print(div)

        emb = np.zeros([max_length, dim])
        emb[:, 0::2] = np.sin(pos / div)
        emb[:, 1::2] = np.cos(pos / div)
        return emb
    

    def add_noise(self, latent, t, noise, max_timestamp):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.line_space(beta_start, beta_end, t, max_timestamp)
        alpha = 1 - betas

        alpha_cum_prod_timestamp = torch.cumprod(alpha, dim=0)
        alpha_bar = alpha_cum_prod_timestamp[t-1]

        noisey_latent = torch.sqrt(1 - alpha_bar) * latent + alpha_bar * noise
        return noisey_latent


    def forward(self, noisy_latent, timsstamp, conditioning):
        # first flatten the latent
        x = noisy_latent.flatten(start_dim=2)

        pos = self.sinusoidal_positional_encoding(x.shape[1], x.shape[2])
        x_pos = x + pos

        x = self.linear_atten(x_pos)
        x = self.cross_atten(x, conditioning)
        x = self.norm(x)
        x_ff = self.mix_ff(x)

        return x_ff
