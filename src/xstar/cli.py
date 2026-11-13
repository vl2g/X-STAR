"""Argument helpers shared by the command-line scripts (no heavy imports)."""

import argparse
import sys
from pathlib import Path


def parser(description):
    return argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)


def add_clip_args(ap):
    g = ap.add_argument_group("CLIP weights")
    g.add_argument("--device", default="cpu", help="torch device (default: cpu). GPUs are used only when named, e.g. cuda:0")
    g.add_argument("--clip-download-root", default=None,
                   help="directory holding ViT-B-32.pt (default: ~/.cache/clip)")
    g.add_argument("--allow-download", action="store_true",
                   help="allow the clip package to download missing ViT-B/32 weights")


def add_limit_arg(ap):
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N items; the output is marked partial (for checks only)")


def add_subset_arg(ap):
    ap.add_argument("--image-ids", default=None,
                    help="file with image IDs (one per line): extract only these records of the split, in split "
                         "order; the output is marked as a subset (partial)")


def fail(message, code=2):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def run(main):
    """Run ``main`` and turn known input errors into a one-line message and exit code 2."""
    from xstar.errors import CheckpointError, ConfigError, DataError, EncoderError, OutputSafetyError

    try:
        return main()
    except (DataError, ConfigError, CheckpointError, OutputSafetyError, EncoderError, FileExistsError,
            FileNotFoundError) as e:
        fail(str(e))


def resolve_checkpoint(value, paths):
    """A checkpoint path, or a key of ``checkpoints:`` in the paths config."""
    if paths is not None and value in paths.checkpoints:
        return paths.checkpoint(value)
    return Path(value)
