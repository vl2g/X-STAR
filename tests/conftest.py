import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

RESEARCH_ROOT = os.environ.get("XSTAR_RESEARCH_ROOT")


@pytest.fixture
def research_root():
    """Local research storage (XSTAR_dataset/, textcap/); tests skip without it."""
    if not RESEARCH_ROOT or not Path(RESEARCH_ROOT, "XSTAR_dataset").is_dir():
        pytest.skip("set XSTAR_RESEARCH_ROOT to the research storage to run this check")
    return Path(RESEARCH_ROOT)


@pytest.fixture(scope="session")
def worker_processes():
    """Fail fast, with a clear reason, if DataLoader workers cannot return batches here.

    Tests that need worker processes use this fixture; in an environment that
    blocks the worker-to-main transfer they report "UNVERIFIED" instead of
    waiting forever (see tools/check_dataloader_workers.py).
    """
    import torch
    from torch.utils.data import DataLoader

    try:
        batch = next(iter(DataLoader([torch.zeros(2)] * 2, batch_size=2, num_workers=1, timeout=30)))
        assert batch.shape == (2, 2)
    except Exception as e:  # environment restriction, not a code failure
        pytest.fail(f"UNVERIFIED in this environment: a one-worker DataLoader returned no batch "
                    f"({type(e).__name__}: {e}); run tools/check_dataloader_workers.py", pytrace=False)
