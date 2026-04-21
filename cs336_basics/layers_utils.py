import torch
import torch.nn as nn

from einops import einsum

from torch import Tensor
from jaxtyping import Float


def silu_activation(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * nn.functional.sigmoid(x)


def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    exponents = torch.exp(x - torch.max(x))
    result = exponents / exponents.sum(dim=dim, keepdim=True)
    return result

def scaled_dot_product_attention(
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Tensor | None = None,
) -> Tensor:
    if mask is None:
        mask = torch.tril(
            torch.ones(Q.shape[-2], K.shape[-1], device=Q.device, dtype=bool)
        )
    result = einsum(Q, K, "... q d_k, ... k d_k -> ... q k").masked_fill(~mask, float("-inf")) / Q.shape[-1] ** 0.5
    scaled_product = einsum(softmax(result), V, "... q k, ... k d_v -> ... q d_v")
    return scaled_product