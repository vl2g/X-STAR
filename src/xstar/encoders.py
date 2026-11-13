"""Frozen CLIP ViT-B/32 encoders, matching the original extraction code.

Uses the original OpenAI ``clip`` package (``pip install
git+https://github.com/openai/CLIP.git``), imported lazily. Weights are read
from ``download_root`` (default ``~/.cache/clip``); a download happens only
when ``allow_download=True``.

* Image tokens: conv1 -> CLS + positional -> ``ln_pre`` -> transformer ->
  ``ln_post`` -> ``proj`` on all 50 tokens. Not L2-normalised.
* Global image: ``encode_image`` then L2 normalisation.
* Query tokens: token + positional embedding -> transformer -> ``ln_final`` ->
  ``text_projection`` on every position, L2-normalised (eps 1e-6); mask is
  ``token_id != 0`` (start, word and end tokens).
* Global text (queries and OCR lines): ``encode_text`` (end-of-text token)
  then L2 normalisation.

Text is tokenised with ``clip.tokenize(..., truncate=True)``.
"""

import math
import os
from pathlib import Path

import numpy as np

from xstar.constants import CLIP_DIM, CLIP_MODEL_NAME, TEXT_CONTEXT
from xstar.errors import EncoderError

# Image loading guards from the original image extractor.
MAX_MEGA_PIXELS = 36.0
MAX_LONG_SIDE = 6000


def _clip():
    try:
        import clip
    except ImportError as e:
        raise EncoderError("feature extraction needs OpenAI CLIP: "
                           "pip install git+https://github.com/openai/CLIP.git") from e
    return clip


def clip_weights_path(download_root=None, name=CLIP_MODEL_NAME):
    clip = _clip()
    root = Path(download_root or os.path.expanduser("~/.cache/clip"))
    url = clip.clip._MODELS[name]
    return root / os.path.basename(url)


def load_clip(device="cpu", download_root=None, allow_download=False, name=CLIP_MODEL_NAME):
    """Load CLIP in float32 eval mode, as the research code did (``jit=False``, ``.float()``)."""

    clip = _clip()
    weights = clip_weights_path(download_root, name)
    if not weights.is_file() and not allow_download:
        raise EncoderError(f"CLIP weights not found at {weights}; pass --allow-download to fetch them "
                           "from the URL built into the clip package, or set --clip-download-root")
    if device.startswith("cuda"):
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            pass
    model, preprocess = clip.load(name, device=device, jit=False, download_root=str(weights.parent))
    return model.float().eval(), preprocess


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def _downscale_needed(w, h):
    return (w * h) > int(MAX_MEGA_PIXELS * 1_000_000) or max(w, h) > MAX_LONG_SIDE


def _scale_target(w, h):
    side_scale = min(1.0, MAX_LONG_SIDE / max(w, h))
    cur_mp = (w * h) / 1_000_000.0
    mp_scale = 1.0 if cur_mp <= 0 else min(1.0, math.sqrt(MAX_MEGA_PIXELS / cur_mp))
    s = min(side_scale, mp_scale)
    if s >= 1.0:
        return (w, h)
    return (max(1, int(round(w * s))), max(1, int(round(h * s))))


def configure_pillow(tolerate_truncated=True):
    """Pillow settings of the original extractor: no pixel-count guard (sizes are
    clamped by ``load_image``) and, by default, tolerance of truncated files."""
    from PIL import Image, ImageFile

    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = bool(tolerate_truncated)


def load_image(path):
    """Open, EXIF-transpose, convert to RGB and clamp very large images (original ``load_image_safe``)."""
    from PIL import Image, ImageOps

    with Image.open(path) as im0:
        fmt = (im0.format or "").upper()
        w, h = im0.size
        if fmt in {"JPEG", "MPO"} and _downscale_needed(w, h):
            try:
                im0.draft("RGB", _scale_target(w, h))
            except Exception:
                pass
        try:
            im = ImageOps.exif_transpose(im0)
        except Exception:
            im = im0
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if _downscale_needed(w, h):
            im = ImageOps.contain(im, _scale_target(w, h), Image.Resampling.LANCZOS)
        return im.copy()


def encode_image_tokens(model, images):
    """``[B, 50, 512]`` post-transformer projected tokens for preprocessed ``[B, 3, 224, 224]``."""
    import torch

    with torch.no_grad():
        return _image_tokens(model, images)


def _image_tokens(model, images):
    import torch

    v = model.visual
    dtype = v.conv1.weight.dtype
    x = v.conv1(images.to(dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = v.class_embedding.to(dtype).unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
    x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(dtype)
    x = v.ln_pre(x).permute(1, 0, 2)
    x = v.transformer(x).permute(1, 0, 2).contiguous()
    if getattr(v, "ln_post", None) is not None:
        x = v.ln_post(x)
    if getattr(v, "proj", None) is not None:
        x = x @ v.proj.to(dtype)
    return x


def encode_image_global(model, images):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        return F.normalize(model.encode_image(images), dim=-1)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def tokenize(texts):
    return _clip().tokenize(list(texts), truncate=True)


def encode_text_tokens(model, texts, device="cpu", batch_size=256):
    """Per-token query embeddings ``[N, 77, 512]`` (L2, eps 1e-6) and masks ``[N, 77]``."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        return _text_tokens(model, texts, device, batch_size, F)


def _text_tokens(model, texts, device, batch_size, F):
    tok_all = np.zeros((len(texts), TEXT_CONTEXT, CLIP_DIM), dtype=np.float32)
    msk_all = np.zeros((len(texts), TEXT_CONTEXT), dtype=bool)
    dtype = model.token_embedding.weight.dtype
    for i in range(0, len(texts), batch_size):
        toks = tokenize(texts[i:i + batch_size]).to(device)
        x = model.token_embedding(toks).to(dtype) + model.positional_embedding.to(dtype)
        x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2).contiguous()
        x = model.ln_final(x) @ model.text_projection.to(dtype)
        x = F.normalize(x, dim=-1, eps=1e-6)
        tok_all[i:i + len(toks)] = x.float().cpu().numpy()
        msk_all[i:i + len(toks)] = (toks != 0).cpu().numpy()
    return tok_all, msk_all


def encode_text_global(model, texts, device="cpu", batch_size=256):
    """Global CLIP text embeddings ``[N, 512]`` (end-of-text token, L2-normalised)."""
    import torch
    import torch.nn.functional as F

    if not texts:
        return np.zeros((0, CLIP_DIM), dtype=np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            toks = tokenize(texts[i:i + batch_size]).to(device)
            out.append(F.normalize(model.encode_text(toks), dim=-1).float().cpu().numpy())
    return np.vstack(out).astype(np.float32)
