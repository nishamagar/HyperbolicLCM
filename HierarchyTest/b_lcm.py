import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class EmbeddingNormalizer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sigma", torch.ones(dim))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sigma.clamp(min=1e-6)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigma + self.mu


class LCMFrontend(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        model_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
        scale_embeddings: bool = True,
        normalizer: Optional[EmbeddingNormalizer] = None,
    ):
        super().__init__()
        self.scale = math.sqrt(model_dim) if scale_embeddings else 1.0
        self.normalizer = normalizer

        self.pre_linear = nn.Linear(embed_dim, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self,
        seqs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Normalize inputs for stability
        if self.normalizer is not None:
            seqs = self.normalizer.normalize(seqs)

        B, T, _ = seqs.shape
        pos = torch.arange(T, device=seqs.device)

        x = self.pre_linear(self.scale * seqs)
        x = x + self.pos_emb(pos)[None, :, :]
        x = self.dropout(x)
        return x, padding_mask


class CausalTransformer(nn.Module):
    def __init__(self, model_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Silence nested tensor warning when supported; fallback for older PyTorch.
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    @staticmethod
    def causal_mask(T: int, device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        T = x.size(1)
        mask = self.causal_mask(T, x.device)
        return self.encoder(x, mask=mask, src_key_padding_mask=padding_mask)


class BaseLCM(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        model_dim: int = 768,
        max_seq_len: int = 64,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.2,
        post_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = model_dim

        self.normalizer = EmbeddingNormalizer(embed_dim)

        self.frontend = LCMFrontend(
            embed_dim=embed_dim,
            model_dim=model_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            scale_embeddings=True,
            normalizer=self.normalizer,
        )

        self.lcm = CausalTransformer(
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.postnet = nn.Linear(model_dim, embed_dim)
        self.post_dropout = nn.Dropout(post_dropout)

    def forward(
        self,
        seqs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        denormalize_output: bool = True,
    ) -> torch.Tensor:
        """
        If denormalize_output=False, returns predictions in normalized space.
        This is recommended for training stability.
        """
        x, padding_mask = self.frontend(seqs, padding_mask)
        h = self.lcm(x, padding_mask)
        y = self.postnet(h)
        y = self.post_dropout(y)

        if denormalize_output:
            return self.normalizer.denormalize(y)
        return y
