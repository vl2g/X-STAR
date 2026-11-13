"""CoLITE-CLIP fusion model.

The classes reproduce the research implementation (main-result notebook cell 1
and the TextCaps pre-training cell) layer for layer, with the same parameter
names, so the original checkpoints load with ``strict=True``.

Temperature: both original checkpoints store ``tau`` = 33.3333, but the two
training sources return different multipliers from ``forward``. That choice is
explicit here through ``tau_mode``:

* ``"fixed"``: return ``fixed_multiplier`` (TextCaps source: 1/0.02 = 50) and
  ignore the stored parameter;
* ``"stored_parameter"``: return the stored ``tau`` parameter (X-STAR source).

``tau`` never receives gradients in either source, and it does not affect
ranking at evaluation time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TAU_MODES = ("fixed", "stored_parameter")


class SoftMHA(nn.Module):
    """Multi-head attention with explicit q/k/v/o projections (source ``SoftMHA``)."""

    def __init__(self, d_model, n_heads, pdrop=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, query, key, value, key_padding_mask=None):
        B, Lq, D = query.shape
        Lk = key.shape[1]
        H, Dh = self.n_heads, self.d_head
        q = self.q_proj(query).view(B, Lq, H, Dh).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, H, Dh).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, H, Dh).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        if key_padding_mask is not None and key_padding_mask.any():
            logits = logits.masked_fill(key_padding_mask.view(B, 1, 1, Lk), float("-inf"))
        attn = self.dropout(torch.softmax(logits, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Lq, D)
        return self.o_proj(out)


class FusionBlock(nn.Module):
    """Pre-LN bidirectional block: V attends to O, then O attends to the updated V."""

    def __init__(self, d, h, pdrop, ffn_mult):
        super().__init__()
        self.attn_vo = SoftMHA(d, h, pdrop)  # V <- O
        self.attn_ov = SoftMHA(d, h, pdrop)  # O <- V
        self.ln_v1 = nn.LayerNorm(d)
        self.ln_v2 = nn.LayerNorm(d)
        self.ln_o1 = nn.LayerNorm(d)
        self.ln_o2 = nn.LayerNorm(d)
        hid = ffn_mult * d
        self.ffn_v = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(pdrop), nn.Linear(hid, d), nn.Dropout(pdrop))
        self.ffn_o = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(pdrop), nn.Linear(hid, d), nn.Dropout(pdrop))

    def forward(self, V, O, mask_V=None, mask_O=None):
        pad_O = None if mask_O is None else (~mask_O)
        V = V + self.attn_vo(self.ln_v1(V), self.ln_o1(O), self.ln_o1(O), key_padding_mask=pad_O)
        V = V + self.ffn_v(self.ln_v2(V))
        O = O + self.attn_ov(self.ln_o1(O), self.ln_v1(V), self.ln_v1(V), key_padding_mask=None)
        O = O + self.ffn_o(self.ln_o2(O))
        return V, O


class FusionLateBank(nn.Module):
    """Fuses visual tokens V and OCR line embeddings O into a normalised token bank.

    Padding OCR slots are replaced by the learned ``null_O`` token and then treated
    as valid, so every image yields ``Nv + K`` bank tokens.
    """

    def __init__(self, d=512, h=4, L=11, pdrop=0.0, ffn_mult=2, tau_mode="stored_parameter",
                 fixed_multiplier=None, init_tau=1 / 0.02):
        super().__init__()
        if tau_mode not in TAU_MODES:
            raise ValueError(f"tau_mode must be one of {TAU_MODES}, got {tau_mode!r}")
        if tau_mode == "fixed" and fixed_multiplier is None:
            raise ValueError("tau_mode='fixed' needs fixed_multiplier")
        self.blocks = nn.ModuleList([FusionBlock(d, h, pdrop, ffn_mult) for _ in range(L)])
        self.null_O = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.tau = nn.Parameter(torch.tensor(float(init_tau)), requires_grad=False)
        self.tau_mode = tau_mode
        self.fixed_multiplier = None if fixed_multiplier is None else float(fixed_multiplier)

    def temperature(self, device=None):
        """Multiplier applied to scores in the InfoNCE loss."""
        if self.tau_mode == "fixed":
            return torch.tensor(self.fixed_multiplier, device=device if device is not None else self.tau.device)
        return self.tau

    def forward(self, V, O, mask_V=None, mask_O=None):
        if mask_O is not None:
            pad_pos = ~mask_O
            if pad_pos.any():
                O = O.clone()
                O[pad_pos] = self.null_O.expand_as(O)[pad_pos]
            mask_O = torch.ones_like(mask_O, dtype=torch.bool, device=O.device)
        for blk in self.blocks:
            V, O = blk(V, O, mask_V, mask_O)
        X = torch.cat([V, O], dim=1)
        mV = mask_V if mask_V is not None else torch.ones(V.size(0), V.size(1), dtype=torch.bool, device=V.device)
        mO = mask_O if mask_O is not None else torch.ones(O.size(0), O.size(1), dtype=torch.bool, device=O.device)
        mX = torch.cat([mV, mO], dim=1)
        X = F.normalize(X, dim=-1, eps=1e-6)
        return X, mX, self.temperature(X.device)
