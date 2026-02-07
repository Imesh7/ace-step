import torch
import torch.nn as nn

class RotationalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.sin_cache = None
        self.cos_cache = None


    def build_cache(self, seq_len, x):
        theta = 1 / 10000 ** (torch.arange(0, self.dim, 2) / self.dim)
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum('i , j -> i j', theta, seq_idx)
        self.sin_cache = torch.sin(idx_theta)
        self.cos_cache = torch.cos(idx_theta)
    
    def neg_pass(self, x):
        d = self.dim // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1, -x2], dim=-1)


    def forward(self, x):
        seq_len = x.shape[1]
        x = self.build_cache(seq_len, x)
        x_rope , x_pass = x[..., :self.dim], x[..., self.dim:]
        
        x_neg = self.neg_pass(x_pass)

        x_res = x_rope * self.cos_cache + x_neg * self.sin_cache
        return torch.cat([x_res, x_pass], dim=-1)
