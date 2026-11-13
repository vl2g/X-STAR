"""YAML configuration: data/feature locations and stage settings.

Two kinds of file are used:

* a **paths** file (see ``configs/paths.example.yaml``) naming every input and
  output location;
* a **stage** file (``configs/textcaps_pretrain.yaml``,
  ``configs/xstar_finetune.yaml``) holding model, temperature and optimisation
  settings for training.

Nothing here touches the file system beyond reading the YAML.
"""

from pathlib import Path

from xstar.constants import QUERY_SETS
from xstar.errors import ConfigError


def load_yaml(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _require(mapping, key, where):
    if not isinstance(mapping, dict) or key not in mapping or mapping[key] is None:
        raise ConfigError(f"missing required setting '{where}.{key}'")
    return mapping[key]


def _path(value):
    return None if value is None else Path(str(value)).expanduser()


class PathsConfig:
    """Resolves the locations of the manifest, split lists, feature files and checkpoints."""

    def __init__(self, data, source="<dict>"):
        self.source = str(source)
        self.raw = data
        dataset = _require(data, "dataset", "")
        self.dataset_root = _path(dataset.get("root"))
        self.manifest = _path(_require(dataset, "manifest", "dataset"))
        self.split_lists = {k: _path(v) for k, v in (dataset.get("split_lists") or {}).items()}

        feats = data.get("features") or {}
        self.features_root = _path(feats.get("root"))
        self.image_tokens_pattern = feats.get("image_tokens", "{split}_image_tokens_postxfmr_vitb32.npy")
        self.image_global_pattern = feats.get("image_global", "{split}_image_global_embeddings.npy")
        self.query_tokens_dir = feats.get("query_tokens_dir", "colbert_packed2")
        self.query_global_english_dir = feats.get("query_global_english_dir", "english_query_vitb32_2")
        self.query_global_to_english_dir = feats.get("query_global_to_english_dir", "to_english_vitb32_2")
        ocr = feats.get("ocr") or {}
        self.ocr_variant = ocr.get("variant", "recognized")
        self.ocr_dirs = ocr.get("dirs") or {"recognized": "ocr_{split}_vitb32_lines"}

        self.checkpoints = {k: _path(v) for k, v in (data.get("checkpoints") or {}).items()}
        self.textcaps = data.get("textcaps") or {}
        self.sources = data.get("sources") or {}
        self.outputs = data.get("outputs") or {}

    @classmethod
    def load(cls, path):
        return cls(load_yaml(path), source=path)

    # ---- dataset -------------------------------------------------------
    def split_list(self, split):
        if split not in self.split_lists or self.split_lists[split] is None:
            raise ConfigError(f"{self.source}: dataset.split_lists.{split} is not set")
        return self.split_lists[split]

    # ---- features ------------------------------------------------------
    def _features(self):
        if self.features_root is None:
            raise ConfigError(f"{self.source}: features.root is not set")
        return self.features_root

    def image_tokens(self, split):
        return self._features() / self.image_tokens_pattern.format(split=split)

    def image_global(self, split):
        return self._features() / self.image_global_pattern.format(split=split)

    def packed_queries(self, split, query_set):
        """(tokens, mask, ids) paths for a packed per-token query set."""
        if query_set not in QUERY_SETS:
            raise ConfigError(f"unknown query set {query_set!r}")
        d = self._features() / self.query_tokens_dir
        suffix = "" if query_set == "english_query" else f"__{query_set}"
        return (d / f"{split}_q_tok{suffix}.npy", d / f"{split}_q_mask{suffix}.npy", d / f"{split}_q_ids{suffix}.txt")

    def global_queries(self, split, query_set):
        """(embeddings, ids, strings) paths for a global CLIP query set."""
        if query_set not in QUERY_SETS:
            raise ConfigError(f"unknown query set {query_set!r}")
        if query_set == "english_query":
            d = self._features() / self.query_global_english_dir
            return (d / f"{split}_english_query_vitb32.npy", d / f"{split}_image_ids.txt",
                    d / f"{split}_english_query_strings.txt")
        d = self._features() / self.query_global_to_english_dir
        return (d / f"{split}_{query_set}_vitb32.npy", d / f"{split}_{query_set}_image_ids.txt",
                d / f"{split}_{query_set}_strings.txt")

    def ocr_dir(self, split, variant=None):
        variant = variant or self.ocr_variant
        if variant not in self.ocr_dirs:
            raise ConfigError(f"{self.source}: OCR variant {variant!r} not in features.ocr.dirs")
        return self._features() / self.ocr_dirs[variant].format(split=split)

    # ---- checkpoints and safety ---------------------------------------
    def checkpoint(self, key):
        if key not in self.checkpoints or self.checkpoints[key] is None:
            raise ConfigError(f"{self.source}: checkpoints.{key} is not set")
        return self.checkpoints[key]

    def protected_roots(self):
        """Input locations that outputs must never overlap."""
        roots = [self.dataset_root, self.manifest, self.features_root]
        roots += list(self.split_lists.values())
        roots += [p.parent for p in self.checkpoints.values() if p is not None]
        for value in _walk_paths(self.textcaps):
            roots.append(Path(value).expanduser().parent)
        for key in ("bstd_detection_root", "rctw17_root"):
            if self.sources.get(key):
                roots.append(Path(self.sources[key]).expanduser())
        return [r for r in roots if r is not None]


def _walk_paths(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk_paths(v)
    elif isinstance(node, str) and ("/" in node or node.endswith((".npy", ".txt", ".json", ".npz"))):
        yield node


class StageConfig:
    """Model, temperature and optimisation settings for one training stage."""

    MODEL_KEYS = ("dim", "heads", "layers", "ffn_mult", "dropout", "ocr_max_lines")
    TRAIN_KEYS = ("epochs", "batch_size", "lr", "weight_decay", "warmup_steps", "grad_clip_norm", "seed")

    def __init__(self, data, source="<dict>"):
        self.source = str(source)
        self.raw = data
        self.stage = _require(data, "stage", "")
        self.data = _require(data, "data", "")
        self.kind = _require(self.data, "kind", "data")
        if self.kind not in ("xstar", "textcaps"):
            raise ConfigError(f"{source}: data.kind must be 'xstar' or 'textcaps'")
        model = _require(data, "model", "")
        self.model = {k: _require(model, k, "model") for k in self.MODEL_KEYS}
        temp = _require(data, "temperature", "")
        self.tau_mode = _require(temp, "mode", "temperature")
        self.fixed_multiplier = temp.get("fixed_multiplier")
        self.init_tau = float(_require(temp, "init_parameter", "temperature"))
        if self.tau_mode == "fixed" and self.fixed_multiplier is None:
            raise ConfigError(f"{source}: temperature.mode 'fixed' needs temperature.fixed_multiplier")
        train = _require(data, "training", "")
        self.training = dict(train)
        for k in self.TRAIN_KEYS:
            _require(train, k, "training")

    @classmethod
    def load(cls, path):
        return cls(load_yaml(path), source=path)

    def model_kwargs(self):
        m = self.model
        return {"d": int(m["dim"]), "h": int(m["heads"]), "L": int(m["layers"]), "pdrop": float(m["dropout"]),
                "ffn_mult": int(m["ffn_mult"]), "tau_mode": self.tau_mode,
                "fixed_multiplier": self.fixed_multiplier, "init_tau": self.init_tau}
