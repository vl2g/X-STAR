"""Training objective: symmetric InfoNCE over in-batch negatives."""

import torch
import torch.nn.functional as F


def symmetric_nce(S, tau):
    """Average of text->image and image->text cross-entropy on ``tau * S``.

    ``S[i, j]`` scores query ``i`` against image ``j``; the diagonal holds the
    positives.
    """
    targets = torch.arange(S.size(0), device=S.device)
    S = S * tau
    return 0.5 * (F.cross_entropy(S, targets) + F.cross_entropy(S.t(), targets))


def multi_caption_nce(score_fn, Q, Q_mask, tau):
    """Mean symmetric InfoNCE over the caption axis.

    ``Q`` is ``[B, C, T, D]`` with C captions per image (C = 5 for TextCaps,
    C = 1 for X-STAR); ``score_fn(Q_c, mask_c)`` returns the ``[B, B]`` scores
    for one caption slot.
    """
    C = Q.shape[1]
    total = 0.0
    for c in range(C):
        total = total + symmetric_nce(score_fn(Q[:, c], Q_mask[:, c]), tau)
    return total / C
