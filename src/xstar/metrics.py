"""Recall@K over a fixed gallery with explicit relevance IDs.

Every query has exactly one relevant gallery item, identified by ID. The rank
of that item is computed directly from the score row, so no gallery row is ever
dropped and no row-order correspondence between queries and images is assumed.

Tie handling (``TIE_POLICY``): items scoring exactly the same as the relevant
item are ordered by their position in the fixed gallery order, i.e. the result
of a stable descending sort over the gallery list. The gallery order is the
order of the split's image list (``dataset.split_lists.<split>`` in the paths
file; for X-STAR, ``image_filenames_test.txt``, which lists the test filenames
sorted). The number of queries whose relevant item is tied with any other item
is reported alongside the recalls. Relevance stays one image per query (its own
record), also when another gallery image is byte-identical; a duplicate-aware
relevance definition would be a different evaluation protocol.
"""

import numpy as np
import torch

TIE_POLICY = "stable_gallery_order"


def relevance_indices(query_ids, gallery_ids):
    """Gallery position of each query's relevant item; raises on unknown IDs."""
    position = {}
    for i, gid in enumerate(gallery_ids):
        if gid in position:
            raise ValueError(f"duplicate gallery ID {gid!r}")
        position[gid] = i
    missing = [q for q in query_ids if q not in position]
    if missing:
        raise KeyError(f"{len(missing)} query IDs have no relevant item in the gallery, e.g. {missing[:5]}")
    return np.array([position[q] for q in query_ids], dtype=np.int64)


def relevant_ranks(scores, relevant):
    """1-based rank of the relevant item per query, and how many queries had ties."""
    scores = torch.as_tensor(scores)
    if scores.dim() != 2:
        raise ValueError(f"scores must be [num_queries, gallery_size], got {tuple(scores.shape)}")
    if torch.isnan(scores).any():
        raise ValueError("scores contain NaN")
    rel = torch.as_tensor(relevant, dtype=torch.long, device=scores.device)
    if rel.shape[0] != scores.shape[0]:
        raise ValueError("one relevant index per query is required")
    target = scores.gather(1, rel[:, None])
    greater = (scores > target).sum(dim=1)
    equal = scores == target
    before = torch.arange(scores.shape[1], device=scores.device)[None, :] < rel[:, None]
    ranks = 1 + greater + (equal & before).sum(dim=1)
    tied_queries = int(((equal.sum(dim=1) - 1) > 0).sum())
    return ranks.cpu().numpy(), tied_queries


def recall_at_k(ranks, ks=(1, 5, 10)):
    """Recall@K in percent from 1-based ranks."""
    ranks = np.asarray(ranks)
    return {f"R@{k}": float((ranks <= k).mean() * 100.0) for k in ks}


def macro_average(per_set, ks=(1, 5, 10)):
    """Unweighted mean of per-query-set recalls."""
    return {f"R@{k}": float(np.mean([m[f"R@{k}"] for m in per_set.values()])) for k in ks}
