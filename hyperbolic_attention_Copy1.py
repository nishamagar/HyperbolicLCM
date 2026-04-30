import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp_tangent(u: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clamp tangent vectors by norm over last dim."""
    if max_norm is None or max_norm <= 0:
        return u
    n = u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (max_norm / n).clamp_max(1.0)
    return u * scale


class HyperbolicMultiHeadAttention(nn.Module):
    """
    Stable "hyperbolic transformer" attention:
      - Treat inputs as points on manifold
      - Logmap0 -> tangent
      - Standard dot-product attention in tangent
      - Output tangent -> expmap0 -> projx (manifold point)

    This avoids expmap(q/k) + hyperbolic dist2 attention, which is often unstable.
    """
    def __init__(
        self,
        manifold,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
        qk_max_norm: float = 2.0,
        out_max_norm: float = 2.0,
        dist2_clip: float = None,  # kept for compatibility; not used
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.manifold = manifold
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.causal = bool(causal)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.qk_max_norm = float(qk_max_norm)
        self.out_max_norm = float(out_max_norm)

    def _safe_expmap0(self, u: torch.Tensor) -> torch.Tensor:
        x = self.manifold.expmap0(u)
        return self.manifold.projx(x)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B,S,D] points on Poincare ball
        attention_mask: [B,S] with 1=keep, 0=mask (optional)
        returns: [B,S,D] points on Poincare ball
        """
        B, S, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        # Map manifold -> tangent at 0
        xt = self.manifold.logmap0(x)  # [B,S,D]

        q = self.q_proj(xt).view(B, S, H, Hd).transpose(1, 2)  # [B,H,S,Hd]
        k = self.k_proj(xt).view(B, S, H, Hd).transpose(1, 2)  # [B,H,S,Hd]
        v = self.v_proj(xt).view(B, S, H, Hd).transpose(1, 2)  # [B,H,S,Hd]

        # Clamp q/k tangent norms to avoid extreme dot products
        q = _clamp_tangent(q, self.qk_max_norm)
        k = _clamp_tangent(k, self.qk_max_norm)

        # Dot-product attention in tangent space
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,S,S]

        # Optional padding mask
        if attention_mask is not None:
            # attention_mask: [B,S] -> [B,1,1,S]
            mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)

        # Optional causal mask
        if self.causal:
            causal = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
            attn_scores = attn_scores.masked_fill(~causal[None, None], torch.finfo(attn_scores.dtype).min)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum in tangent space
        out_tan = torch.matmul(attn_probs, v)  # [B,H,S,Hd]
        out_tan = out_tan.transpose(1, 2).reshape(B, S, D)  # [B,S,D]
        out_tan = self.out_proj(out_tan)

        # Clamp and map back to manifold
        out_tan = _clamp_tangent(out_tan, self.out_max_norm)
        return self._safe_expmap0(out_tan)