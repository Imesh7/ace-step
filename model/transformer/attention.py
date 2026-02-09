from model.RoPE import RotationalPositionalEmbedding
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.d_model = in_channels
        self.head_dim = in_channels // num_heads
        # multiplied by 3 , beacuse we have 'q' , 'k' & 'v' seperations
        self.qkv = nn.Linear(in_features=in_channels, out_features=3 * out_channels)
        self.proj = nn.Linear(in_features=out_channels, out_features=out_channels)

    def forward(self, x):
        batch_size, samples, emb_dim = x.shape
        # In here we have a Tensor Like [5, 12, 3*512]
        # this is like 5 batch, 12 samples, 3 * 512 (dimentios)
        qkv = self.qkv(x)

        # Lets reshape it
        qkv = qkv.view(batch_size, samples, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv

        mask = torch.tril(torch.ones(samples, self.head_dim))

        x = F.scaled_dot_product_attention(query, key, value, atten_mask=mask)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, samples, self.out_channels)
        proj = self.proj(x)
        return proj



class LinearAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.d_model = in_channels
        self.head_dim = in_channels // num_heads
        # multiplied by 3 , beacuse we have 'q' , 'k' & 'v' seperations
        self.q = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.k = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.v = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.proj = nn.Linear(in_features=out_channels, out_features=out_channels)


    def elu_feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        batch_size, samples, emb_dim = x.shape
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        query = self.elu_feature_map(query)
        key = self.elu_feature_map(key)
        # Compute the attention scores
        kv = torch.einsum('bshd,bshe->bhde', key, value)


        z = 1 / torch.einsum('bshd,bhd->bsh', query, key.sum(dim=1))
        output = torch.einsum('bshd,bhde,bsh->bshe', query, kv, z)
        output = self.proj(output)
        return output