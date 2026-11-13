#!/usr/bin/env python3
"""Check whether PyTorch DataLoader worker processes can return batches here.

A DataLoader worker passes each batch to the main process through shared
memory. If the environment denies that (some sandboxes restrict /dev/shm or
shm_open), the error is raised in the worker queue's background thread, the
batch is dropped, and a loader without a timeout waits forever. The trainer
therefore uses a batch timeout (training.dataloader_timeout_s).

Checks, each bounded in time:
  1. /dev/shm exists and is writable;
  2. a tensor can be moved to shared memory in this process;
  3. a DataLoader with one worker returns a batch within --timeout seconds.

--simulate-denied-shm makes step 3 deny shared memory inside the worker, to
show the failure mode: with a timeout it ends in a clear error.

Exit status 0 when workers work, 1 otherwise. Needs only PyTorch.
"""

import argparse
import os
import sys
import tempfile
import time


def _deny_shm(_worker_id):
    import torch

    def refuse(self, *a, **k):
        raise OSError(1, "simulated: shared memory denied")
    torch.UntypedStorage._share_fd_cpu_ = refuse
    torch.UntypedStorage._share_filename_cpu_ = refuse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for the worker batch")
    ap.add_argument("--simulate-denied-shm", action="store_true", help="deny shared memory inside the worker")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset

    ok = True
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"start method {torch.multiprocessing.get_start_method()}, sharing strategy "
          f"{torch.multiprocessing.get_sharing_strategy()}")
    try:
        with tempfile.NamedTemporaryFile(dir="/dev/shm"):
            pass
        print("1. /dev/shm writable: yes")
    except OSError as e:
        ok = False
        print(f"1. /dev/shm writable: NO ({e})")
    try:
        torch.zeros(4).share_memory_()
        print("2. tensor.share_memory_() in this process: yes")
    except Exception as e:
        ok = False
        print(f"2. tensor.share_memory_() in this process: NO ({type(e).__name__}: {e})")

    class Tiny(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return torch.full((3,), float(i))

    loader = DataLoader(Tiny(), batch_size=2, num_workers=1, timeout=args.timeout,
                        worker_init_fn=_deny_shm if args.simulate_denied_shm else None)
    start = time.time()
    try:
        batch = next(iter(loader))
        print(f"3. one-worker DataLoader batch: yes ({tuple(batch.shape)}, {time.time() - start:.1f}s, pid {os.getpid()})")
    except Exception as e:
        ok = False
        print(f"3. one-worker DataLoader batch: NO after {time.time() - start:.1f}s ({type(e).__name__}: {e})")
    print("RESULT:", "worker processes usable" if ok else "worker processes NOT usable here; use --num-workers 0")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
