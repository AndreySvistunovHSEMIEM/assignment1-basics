import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum

from layers_utils import sil


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
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3 * std, b=3 * std)

        self.weight = nn.Parameter(weight)
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
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
    
    def forward(self, token_ids: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_a = torch.sqrt(1 / self.d_model * torch.sum(x * x, dim=-1, keepdim=True) + self.eps)
        result = x * self.weight / rms_a
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int | None = None,
            device: torch.device = None,
            dtype: torch.dtype = None
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int(round(8 * d_model / 3))
        
        self.w1 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w2 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        x_w1 = silu_activation(self.w1(x))
        x_w3 = self.w3(x)
        x_w1_w3 = x_w1 * x_w3
        return self.w2(x_w1_w3)


class RoPE(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even!"
        
        k_elements = torch.arange(start=0, end=(d_k // 2), device=device)
        positions = torch.arange(start=0, end=max_seq_len, device=device).unsqueeze(-1)
        angles = positions / theta ** (2 * k_elements / d_k)

        sin_elements = torch.sin(angles)
        cos_elements = torch.cos(angles)

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.register_buffer("sin_elements", sin_elements, persistent=False)
        self.register_buffer("cos_elements", cos_elements, persistent=False)


    def forward(
            self,
            x: Float[Tensor, "... seq_len d_k"],
            token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        
        result = torch.empty(x.size(), device=x.device, dtype=x.dtype)

        even_part = x[..., torch.arange(start=0, end=self.d_k, step=2)]
        odd_part = x[..., torch.arange(start=1, end=self.d_k, step=2)]

        even_part, odd_part  = (
            even_part * self.cos_elements[token_positions] - odd_part * self.sin_elements[token_positions],
            even_part * self.sin_elements[token_positions] + odd_part * self.cos_elements[token_positions],
        )

        result[..., 0::2] = even_part
        result[..., 1::2] = odd_part
        
        return result