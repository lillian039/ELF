import inspect
import logging
import os


def _process_index() -> int:
    """Return torch.distributed rank, falling back to env vars or 0."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def log_for_0(msg, *args, level=logging.INFO):
    """Log only on the first process (rank == 0)."""
    if _process_index() != 0:
        return
    caller_module = inspect.currentframe().f_back.f_globals.get("__name__", __name__)
    logging.getLogger(caller_module).log(level, msg, *args)
