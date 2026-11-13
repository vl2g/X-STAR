"""Original research implementations, kept only as a test oracle.

Copied from the recovered sources with module-level constants turned into
arguments; the computations are unchanged:

* ``XStarFusionLateBank``, ``maxsim_scores_batch``, ``symmetric_nce``:
  main-result notebook ``XSTAR_finetune_5(MAIN_RESULT).ipynb`` cell 1.
* ``EvalFusionLateBank``: the evaluator copy in cell 12 of the same notebook
  (``mask_O`` reset inside the padding branch).
* ``TextCapsFusionLateBank``: server-40 ``ColBERT_style_new_13(main).ipynb`` cell 2.

``SoftMHA`` and ``FusionBlock`` are identical in all three sources.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMHA(nn.Module):
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
            pad = key_padding_mask.view(B, 1, 1, Lk)
            logits = logits.masked_fill(pad, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Lq, D)
        return self.o_proj(out)


class FusionBlock(nn.Module):
    def __init__(self, d, h, pdrop, ffn_mult):
        super().__init__()
        self.attn_vo = SoftMHA(d, h, pdrop)
        self.attn_ov = SoftMHA(d, h, pdrop)
        self.ln_v1 = nn.LayerNorm(d); self.ln_v2 = nn.LayerNorm(d)
        self.ln_o1 = nn.LayerNorm(d); self.ln_o2 = nn.LayerNorm(d)
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


class XStarFusionLateBank(nn.Module):
    def __init__(self, d, h, L, pdrop, ffn_mult):
        super().__init__()
        self.blocks = nn.ModuleList([FusionBlock(d, h, pdrop, ffn_mult) for _ in range(L)])
        self.null_O = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.tau = nn.Parameter(torch.tensor(1 / 0.02), requires_grad=False)

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
        return X, mX, self.tau


class EvalFusionLateBank(nn.Module):
    def __init__(self, d, h, L, pdrop, ffn_mult):
        super().__init__()
        self.blocks = nn.ModuleList([FusionBlock(d, h, pdrop, ffn_mult) for _ in range(L)])
        self.null_O = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.tau = nn.Parameter(torch.tensor(1 / 0.02))
        self.learn_tau = False

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
        return X, mX, (self.tau if self.learn_tau else torch.tensor(1 / 0.02, device=X.device))


class TextCapsFusionLateBank(nn.Module):
    def __init__(self, d, h, L, pdrop, ffn_mult):
        super().__init__()
        self.blocks = nn.ModuleList([FusionBlock(d, h, pdrop, ffn_mult) for _ in range(L)])
        self.null_O = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.tau = nn.Parameter(torch.tensor(1 / 0.03))
        self.learn_tau = False

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
        return X, mX, (self.tau if self.learn_tau else torch.tensor(1 / 0.02, device=X.device))


def maxsim_scores_batch(Q_tok, Q_mask, X_bank, X_mask, chunk_b=64, l2norm_q=True):
    B, T, _ = Q_tok.shape
    if l2norm_q:
        Q_tok = F.normalize(Q_tok, dim=-1, eps=1e-6)
    S = torch.zeros(B, B, device=Q_tok.device, dtype=Q_tok.dtype)
    Q_slices = []
    for i in range(B):
        mi = Q_mask[i]
        Qi = Q_tok[i][mi] if mi.any() else Q_tok[i][:0]
        Q_slices.append(Qi)
    for j0 in range(0, B, chunk_b):
        j1 = min(B, j0 + chunk_b)
        Xj = X_bank[j0:j1]
        Mj = X_mask[j0:j1]
        X_slices = []
        for jj in range(j1 - j0):
            mj = Mj[jj]
            Xs = Xj[jj][mj] if mj.any() else Xj[jj][:0]
            X_slices.append(Xs)
        for i in range(B):
            Qi = Q_slices[i]
            if Qi.numel() == 0:
                continue
            for jj in range(j1 - j0):
                Xs = X_slices[jj]
                if Xs.numel() == 0:
                    continue
                sims = Qi @ Xs.t()
                s = sims.max(dim=1).values.sum()
                S[i, j0 + jj] = s
    return S


def symmetric_nce(S, tau):
    targets = torch.arange(S.size(0), device=S.device)
    S = S * tau
    return 0.5 * (F.cross_entropy(S, targets) + F.cross_entropy(S.t(), targets))
