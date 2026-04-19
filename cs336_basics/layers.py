import torch
import torch.nn as nn
from einops import einsum
import math


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None =None,
    ) -> None:
        super().__init__()

        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3 * std, b=3 * std)

        self.weight = nn.Parameter(weight)
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_features, "Dimension error!"
        result = einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
        return result


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, std=1, a=-3, b=3)

        self.weight = nn.Parameter(weight)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None =None,
            dtype: torch.dtype | None =None,
        ) -> None:
        super().__init__()

        weight = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_a = torch.sqrt(1 / self.d_model * torch.sum(x * x, dim=-1, keepdim=True) + self.eps)
        result = x * self.weight / rms_a
        return result.to(in_dtype)