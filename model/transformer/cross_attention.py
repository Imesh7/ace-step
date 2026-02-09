import torch
import torch.nn as nn
import torch.nn.functional as F
from model.RoPE import RotationalPositionalEmbedding


# Implement cross attention for the TAGS, LY
class CrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, rope_enabled=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # This will RoPE or Linear projection for query and key, depending on the rope_enabled flag
        if rope_enabled:
            self.w_q = RotationalPositionalEmbedding(dim=self.in_channels)
            self.w_k = RotationalPositionalEmbedding(dim=self.in_channels)
        else:
            self.w_q = nn.Linear(
                in_features=self.in_channels, out_features=self.out_channels
            )
            self.w_k = nn.Linear(
                in_features=self.in_channels, out_features=self.out_channels
            )

        self.w_v = nn.Linear(
            in_features=self.in_channels, out_features=self.out_channels
        )
        self.proj = nn.Linear(
            in_features=self.out_channels, out_features=self.out_channels
        )

    def forward(self, source, target):
        batch_size, samples, emb_dim = source.shape

        # This is cross Attention's important part
        # query will poject with input source
        query = self.w_q(source)

        # key & value will poject with target source (External source)
        key = self.w_k(target)
        value = self.w_v(target)

        mask = torch.tril(torch.ones(samples, self.head_dim))

        x = F.scaled_dot_product_attention(query, key, value, atten_mask=mask)

        proj = self.proj(x)
        return proj
