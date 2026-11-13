"""Late-interaction (MaxSim) scoring and the CLIP global baseline.

The research code computed MaxSim with Python loops over every query/image
pair. ``maxsim_scores`` computes the same quantity in chunks:

    score(q, g) = sum over valid query tokens t of  max over valid bank tokens m of  <Q[q,t], X[g,m]>

Pairs where the image has no valid bank token, or the query has no valid token,
score 0, as in the original evaluator. The original sums only valid tokens
while this version masks invalid tokens to zero before summing, so the two can
differ in the last bits of float32 summation order.
"""

import torch
import torch.nn.functional as F


def maxsim_reference(Q, Q_mask, X, X_mask):
    """Direct per-pair MaxSim (the original loop formulation). For tests and checks."""
    S = torch.zeros(Q.shape[0], X.shape[0], dtype=Q.dtype, device=Q.device)
    for i in range(Q.shape[0]):
        Qi = Q[i][Q_mask[i]]
        if Qi.numel() == 0:
            continue
        for j in range(X.shape[0]):
            Xj = X[j][X_mask[j]]
            if Xj.numel() == 0:
                continue
            S[i, j] = (Qi @ Xj.t()).max(dim=1).values.sum()
    return S


def maxsim_scores(Q, Q_mask, X, X_mask, query_chunk=32, gallery_chunk=128, normalize_queries=False):
    """MaxSim scores for every query/gallery pair.

    Args:
        Q: query token embeddings ``[Nq, T, D]``.
        Q_mask: valid query tokens ``[Nq, T]`` (bool).
        X: fused gallery bank ``[Ng, M, D]``.
        X_mask: valid bank tokens ``[Ng, M]`` (bool).
        query_chunk, gallery_chunk: chunk sizes bounding the
            ``query_chunk * gallery_chunk * T * M`` similarity buffer.
        normalize_queries: L2-normalise query tokens first (eps 1e-6). The
            X-STAR training source does this; the evaluators do not.

    Returns:
        ``[Nq, Ng]`` scores. Differentiable, so also used for in-batch training.
    """
    if Q.dim() != 3 or X.dim() != 3:
        raise ValueError(f"expected Q [Nq,T,D] and X [Ng,M,D], got {tuple(Q.shape)} and {tuple(X.shape)}")
    if Q.shape[-1] != X.shape[-1]:
        raise ValueError(f"embedding size mismatch: {Q.shape[-1]} vs {X.shape[-1]}")
    if normalize_queries:
        Q = F.normalize(Q, dim=-1, eps=1e-6)
    Q_mask = Q_mask.bool()
    X_mask = X_mask.bool()
    rows = []
    for q0 in range(0, Q.shape[0], query_chunk):
        Qc, Qmc = Q[q0:q0 + query_chunk], Q_mask[q0:q0 + query_chunk]
        cols = []
        for g0 in range(0, X.shape[0], gallery_chunk):
            Xc, Xmc = X[g0:g0 + gallery_chunk], X_mask[g0:g0 + gallery_chunk]
            sims = torch.einsum("qtd,gmd->qgtm", Qc, Xc)
            sims = sims.masked_fill(~Xmc[None, :, None, :], float("-inf"))
            best = sims.max(dim=-1).values                                   # [q, g, t]
            keep = Qmc[:, None, :] & Xmc.any(dim=-1)[None, :, None]
            best = torch.where(keep, best, torch.zeros((), dtype=best.dtype, device=best.device))
            cols.append(best.sum(dim=-1))
        rows.append(torch.cat(cols, dim=1))
    return torch.cat(rows, dim=0)


def clip_global_scores(query_global, image_global):
    """Cosine similarity between global CLIP query and image embeddings ``[Nq, Ng]``."""
    q = F.normalize(torch.as_tensor(query_global, dtype=torch.float32), dim=-1)
    g = F.normalize(torch.as_tensor(image_global, dtype=torch.float32), dim=-1)
    return q @ g.t()
