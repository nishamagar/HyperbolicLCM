import torch
import torch.nn as nn
from geoopt.manifolds import PoincareBall
from hyperbolic_attention import HyperbolicMultiHeadAttention


def _clamp_tangent(u: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm is None or max_norm <= 0:
        return u
    n = u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (max_norm / n).clamp_max(1.0)
    return u * scale


class TangentLayerNorm(nn.Module):
    def __init__(self, dim, manifold, eps=1e-5):
        super().__init__()
        self.manifold = manifold
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        xt = self.manifold.logmap0(x)
        xt = self.ln(xt)
        xh = self.manifold.expmap0(xt)
        return self.manifold.projx(xh)


class HyperbolicFeedForward(nn.Module):
    def __init__(self, model_dim, manifold, ffn_mult=4, dropout=0.1, out_max_norm: float = 2.0):
        super().__init__()
        self.manifold = manifold
        hidden = int(ffn_mult) * int(model_dim)
        self.fc1 = nn.Linear(model_dim, hidden)
        self.fc2 = nn.Linear(hidden, model_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_max_norm = float(out_max_norm)

    def forward(self, x):
        xt = self.manifold.logmap0(x)
        out = self.fc2(self.dropout(self.act(self.fc1(xt))))
        out = _clamp_tangent(out, self.out_max_norm)
        xh = self.manifold.expmap0(out)
        return self.manifold.projx(xh)


class HyperbolicTransformerLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        num_heads,
        manifold,
        ffn_mult=4,
        dropout=0.1,
        causal=False,
        res_max_norm: float = 2.0,
        attn_qk_max_norm: float = 2.0,
        attn_out_max_norm: float = 2.0,
        ff_out_max_norm: float = 2.0,
    ):
        super().__init__()
        self.manifold = manifold

        self.attn = HyperbolicMultiHeadAttention(
            manifold=manifold,
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
            qk_max_norm=attn_qk_max_norm,
            out_max_norm=attn_out_max_norm,
        )

        self.ff = HyperbolicFeedForward(
            model_dim=model_dim,
            manifold=manifold,
            ffn_mult=ffn_mult,
            dropout=dropout,
            out_max_norm=ff_out_max_norm,
        )

        self.norm1 = TangentLayerNorm(model_dim, manifold)
        self.norm2 = TangentLayerNorm(model_dim, manifold)
        self.dropout = nn.Dropout(dropout)
        self.res_max_norm = float(res_max_norm)

    def _safe_expmap0(self, u: torch.Tensor) -> torch.Tensor:
        x = self.manifold.expmap0(u)
        return self.manifold.projx(x)

    def forward(self, x):
        # Attention block
        h = self.norm1(x)
        a = self.attn(h)  # manifold point

        da_tan = self.manifold.logmap0(a)
        da_tan = self.dropout(da_tan)
        da_tan = _clamp_tangent(da_tan, self.res_max_norm)
        da = self._safe_expmap0(da_tan)

        x = self.manifold.mobius_add(x, da)
        x = self.manifold.projx(x)

        # FFN block
        h = self.norm2(x)
        f = self.ff(h)  # manifold point

        df_tan = self.manifold.logmap0(f)
        df_tan = self.dropout(df_tan)
        df_tan = _clamp_tangent(df_tan, self.res_max_norm)
        df = self._safe_expmap0(df_tan)

        x = self.manifold.mobius_add(x, df)
        x = self.manifold.projx(x)
        return x


class HyperbolicLCM(nn.Module):
    def __init__(
        self,
        in_dim=768,
        model_dim=1024,
        num_heads=16,
        num_layers=8,
        ffn_mult=4,
        dropout=0.1,
        manifold_c=0.01,
        init_scale=0.02,
        causal=False,
        input_scale=0.05,
        input_max_norm=1.0,
    ):
        super().__init__()
        self.manifold = PoincareBall(c=manifold_c)

        # Projection in Euclidean tangent space at 0
        self.input_proj = nn.Linear(in_dim, model_dim)

        nn.init.normal_(self.input_proj.weight, std=init_scale)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        self.layers = nn.ModuleList([
            HyperbolicTransformerLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                manifold=self.manifold,
                ffn_mult=ffn_mult,
                dropout=dropout,
                causal=causal,
                res_max_norm=2.0,
                attn_qk_max_norm=2.0,
                attn_out_max_norm=2.0,
                ff_out_max_norm=2.0,
            )
            for _ in range(num_layers)
        ])

        self.input_scale = float(input_scale)
        self.input_max_norm = float(input_max_norm)

    def _safe_expmap0(self, u: torch.Tensor) -> torch.Tensor:
        x = self.manifold.expmap0(u)
        return self.manifold.projx(x)

    def encode_inputs(self, x):
        xt = self.input_proj(x)
        xt = _clamp_tangent(xt * self.input_scale, self.input_max_norm)
        return self._safe_expmap0(xt)

    def encode_full(self, x):
        """Full encoder: proj->manifold then transformer layers."""
        xh = self.encode_inputs(x)
        for layer in self.layers:
            xh = layer(xh)
        return xh

    def forward(self, x):
        return self.encode_full(x)
