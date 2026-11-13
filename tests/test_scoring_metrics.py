import numpy as np
import pytest
import torch

from xstar.metrics import macro_average, recall_at_k, relevance_indices, relevant_ranks
from xstar.scoring import clip_global_scores, maxsim_reference, maxsim_scores


def _random(Nq=7, Ng=9, T=6, M=5, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.randn(Nq, T, D, generator=g)
    X = torch.randn(Ng, M, D, generator=g)
    Qm = torch.rand(Nq, T, generator=g) > 0.4
    Xm = torch.rand(Ng, M, generator=g) > 0.3
    Qm[0] = False        # query without valid tokens
    Xm[1] = False        # gallery item without valid tokens
    Qm[2] = True
    return Q, Qm, X, Xm


def test_chunked_maxsim_matches_reference_with_masks():
    Q, Qm, X, Xm = _random()
    ref = maxsim_reference(Q, Qm, X, Xm)
    assert torch.all(ref[0] == 0) and torch.all(ref[:, 1] == 0)
    for qc, gc in [(1, 1), (2, 3), (7, 9), (64, 128)]:
        torch.testing.assert_close(maxsim_scores(Q, Qm, X, Xm, query_chunk=qc, gallery_chunk=gc), ref,
                                   rtol=1e-6, atol=1e-6)


def test_padding_tokens_do_not_change_scores():
    Q, Qm, X, Xm = _random(seed=1)
    base = maxsim_scores(Q, Qm, X, Xm)
    Q2, X2 = Q.clone(), X.clone()
    Q2[~Qm] = 1e3            # garbage in masked query positions
    X2[~Xm] = 1e3            # garbage in masked bank positions
    torch.testing.assert_close(maxsim_scores(Q2, Qm, X2, Xm), base)


def test_clip_global_scores_normalise():
    q, g = np.array([[3.0, 0.0]]), np.array([[0.0, 2.0], [5.0, 0.0]])
    torch.testing.assert_close(clip_global_scores(q, g), torch.tensor([[0.0, 1.0]]))


def test_recall_uses_explicit_ids_with_shuffled_gallery():
    gallery = ["c", "a", "d", "b"]
    queries = ["a", "b", "c"]           # not in gallery order
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.3],    # a ranked 1st
                           [0.5, 0.1, 0.4, 0.3],    # b ranked 3rd
                           [0.0, 0.2, 0.9, 0.8]])   # c ranked 4th
    rel = relevance_indices(queries, gallery)
    ranks, ties = relevant_ranks(scores, rel)
    assert ranks.tolist() == [1, 3, 4] and ties == 0
    assert recall_at_k(ranks, (1, 3)) == {"R@1": pytest.approx(100 / 3), "R@3": pytest.approx(200 / 3)}


def test_queries_subset_never_shrinks_gallery():
    gallery = [f"g{i}" for i in range(6)]
    scores = torch.zeros(2, 6)
    scores[0, 4] = 1.0
    scores[1, 5] = 1.0
    ranks, _ = relevant_ranks(scores, relevance_indices(["g4", "g5"], gallery))
    assert ranks.tolist() == [1, 1]
    with pytest.raises(KeyError):
        relevance_indices(["missing"], gallery)
    with pytest.raises(ValueError):
        relevance_indices(["g1"], gallery + ["g1"])


def test_ties_are_deterministic_and_follow_gallery_order():
    scores = torch.tensor([[0.5, 0.5, 0.5, 0.1]])
    assert relevant_ranks(scores, [0])[0].tolist() == [1]
    r2, ties = relevant_ranks(scores, [2])
    assert r2.tolist() == [3] and ties == 1
    # Same result as a stable descending sort over the gallery order.
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices[0].tolist()
    assert order.index(2) + 1 == 3
    with pytest.raises(ValueError):
        relevant_ranks(torch.tensor([[float("nan"), 0.0]]), [0])


def test_macro_average():
    per = {"a": {"R@1": 10.0, "R@5": 20.0, "R@10": 30.0}, "b": {"R@1": 30.0, "R@5": 40.0, "R@10": 50.0}}
    assert macro_average(per) == {"R@1": 20.0, "R@5": 30.0, "R@10": 40.0}
