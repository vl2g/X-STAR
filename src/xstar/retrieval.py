"""Gallery token banks and ranked retrieval with a CoLITE-CLIP checkpoint."""

import json
from pathlib import Path

import numpy as np
import torch

from xstar.errors import DataError
from xstar.scoring import maxsim_scores


def build_bank(model, images, ids, ocr_bank, batch_size=256, device="cpu", null_visual=False):
    """Fuse visual tokens and OCR lines for ``ids`` into ``X [N, Nv+K, D]`` and ``mask [N, Nv+K]``.

    ``null_visual=True`` feeds zeros instead of the visual tokens (the
    null-visual ablation of main-result notebook cell 14).
    """
    rows = images.rows_for(ids)
    model.eval()
    X_parts, M_parts = [], []
    with torch.no_grad():
        for b0 in range(0, len(ids), batch_size):
            b_ids = ids[b0:b0 + batch_size]
            V = torch.from_numpy(np.stack([images.get(int(r)) for r in rows[b0:b0 + batch_size]]))
            O, mO = zip(*(ocr_bank.get(i) for i in b_ids))
            O = torch.from_numpy(np.stack(O))
            mO = torch.from_numpy(np.stack(mO))
            if null_visual:
                V = torch.zeros_like(V)
            X, mX, _ = model(V.to(device), O.to(device), mask_V=None, mask_O=mO.to(device))
            X_parts.append(X.cpu())
            M_parts.append(mX.cpu())
    return torch.cat(X_parts), torch.cat(M_parts)


def save_bank(path, X, mask, ids, checkpoint_sha256, details):
    """Cache a fused bank (``.npz``); refuses to overwrite."""
    path = Path(path)
    with open(path, "xb") as f:
        np.savez(f, X=X.numpy().astype(np.float32), mask=mask.numpy(), ids=np.array(ids, dtype=np.str_),
                 checkpoint_sha256=np.array(checkpoint_sha256), details=np.array(json.dumps(details, sort_keys=True)))
    return path


def load_bank(path, checkpoint_sha256, expected_ids=None, expected_details=None):
    """Load a cached bank, checking checkpoint, gallery IDs and build settings (split, K, OCR source)."""
    with np.load(path, allow_pickle=False) as z:
        sha = str(z["checkpoint_sha256"])
        if sha != checkpoint_sha256:
            raise DataError(f"{path} was built from checkpoint {sha[:12]}, not {checkpoint_sha256[:12]}")
        ids = [str(x) for x in z["ids"].tolist()]
        if expected_ids is not None and ids != list(expected_ids):
            raise DataError(f"{path}: gallery IDs differ from the requested gallery")
        details = json.loads(str(z["details"]))
        if expected_details is not None and details != json.loads(json.dumps(expected_details, sort_keys=True)):
            raise DataError(f"{path}: built with {details}, not {expected_details}")
        return torch.from_numpy(z["X"]), torch.from_numpy(z["mask"]), ids


def rank(Q, Q_mask, X, X_mask, top_k=10, device="cpu", query_chunk=32, gallery_chunk=128):
    """Top-K gallery positions and scores per query; ties keep gallery order.

    The query tokens and mask (NumPy arrays or tensors) and the gallery bank
    and mask are all placed on ``device`` before scoring, so free-text and
    cached queries work against a bank on any device. Dtypes are unchanged.
    """
    dev = torch.device(device)
    Q, Q_mask, X, X_mask = (torch.as_tensor(a).to(dev) for a in (Q, Q_mask, X, X_mask))
    S = maxsim_scores(Q, Q_mask, X, X_mask, query_chunk=query_chunk, gallery_chunk=gallery_chunk)
    scores, order = torch.sort(S, dim=1, descending=True, stable=True)
    return order[:, :top_k], scores[:, :top_k]
