import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum, rearrange

from .layers_utils import silu_activation, scaled_dot_product_attention


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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
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
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even!"

        positions = torch.arange(0, max_seq_len, 1, device=device, dtype=dtype).unsqueeze(-1)
        k_numbers = torch.arange(0, d_k // 2, 1, device=device, dtype=dtype)

        angles = positions / theta ** (2 * k_numbers / d_k)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        
        even_part = x[..., 0::2]
        odd_part = x[..., 1::2]

        even_part, odd_part = (
            even_part * self.cos_cached[token_positions] - odd_part * self.sin_cached[token_positions],
            odd_part * self.cos_cached[token_positions] + even_part * self.sin_cached[token_positions],
        )

        x[..., 0::2] = even_part
        x[..., 1::2] = odd_part

        return x


class MultiheadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            theta: float | None = None,
            max_seq_len: int | None = None,
            dtype: torch.dtype | None = None,
            device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads!"

        self.d_k = d_model // num_heads
        self.rope = None
        if theta and max_seq_len:
            self.rope = RoPE(theta, self.d_k, max_seq_len, dtype, device)
        
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

        self.device = device

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Tensor:
        Q = rearrange(self.q_proj(x), "... s (a b) -> ... a s b", b=self.d_k)
        K = rearrange(self.k_proj(x), "... s (a b) -> ... a s b", b=self.d_k)
        V = rearrange(self.v_proj(x), "... s (a b) -> ... a s b", b=self.d_k)
        if self.rope:
            if token_positions is None:
                token_positions = torch.arange(0, x.shape[-2], step=1, device=self.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        result = self.output_proj(
            rearrange(scaled_dot_product_attention(Q, K, V), "a b c d -> a c (b d)")
        )
        return result


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            eps: float = 0.00001,
            max_seq_len: int | None = None,
            theta: float | None = None,
            dtype: torch.dtype | None = None,
            device: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.attn = MultiheadAttention(d_model, num_heads, theta, max_seq_len, dtype, device)
        
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        self.ffn = SwiGLU(d_ff, d_model, device, dtype)

        self.device = device

    def forward(
            self,
            x: Tensor,
            token_positions: Tensor | None = None
    ) -> Tensor:
        if token_positions is not None:
            token_positions = torch.arange(x.shape[-2], device=self.device)
        mha = x + self.attn(self.ln1(x), token_positions)
        result = mha + self.ffn(self.ln2(mha))
        return result


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            rms_eps: float = 0.00001,
            dtype: torch.dtype | None = None,
            device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rms_eps, context_length, rope_theta, dtype, device)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, rms_eps, device, dtype)

        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(
            self,
            in_indices: Int[Tensor, " batch_size sequence_length"],
            token_positions: Tensor | None = None,
    ) -> Tensor:
        activations = self.token_embeddings(in_indices)
        for layer in self.layers:
            activations = layer(activations, token_positions)
        activations = self.ln_final(activations)
        activations = self.lm_head(activations)
        return activations
