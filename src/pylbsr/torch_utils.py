"""GPU/device selection and reproducibility helpers for torch."""

import logging
import os

try:
    import torch
except ImportError as e:
    raise ImportError(
        "torch is required for pylbsr.torch_utils. Install with `pip install pylbsr[torch]`."
    ) from e

from pylbsr.misc import set_seed as _set_seed_generic

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def set_seed(seed: int = 42) -> None:
    """Seed Python's `random`, numpy, and torch (CPU + all CUDA devices) for reproducibility.

    Calls `pylbsr.misc.set_seed` first (random + numpy), then seeds torch and configures
    cuDNN/torch for deterministic algorithm selection, at some performance cost -- appropriate
    for reproducible experiments, not tuned for training throughput. If you don't need torch,
    use `pylbsr.misc.set_seed` directly instead of pulling in this module's torch dependency.

    Args:
        seed: Seed value applied to every RNG.
    """
    # Required by cuBLAS when torch.use_deterministic_algorithms(True) is active
    # on CUDA < 12.x; harmless on newer stacks. Must be set before cuBLAS init.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _set_seed_generic(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if CUDA isn't available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    logger.info("Seed set to %d (+ torch/CUDA/cuDNN)", seed)


dtype_str_map = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


def select_gpu(query: int | str) -> torch.device:
    """Resolve a GPU by index, "cuda:N" string, or device-name substring.

    Examples:
    --------
    select_gpu(0)            # first GPU by index
    select_gpu("cuda:1")     # explicit CUDA string
    select_gpu("RTX 5090")   # first GPU whose name contains "RTX 5090"
    select_gpu("5090")       # same, case-insensitive substring match

    Sets torch.cuda.set_device() on the resolved GPU and returns the
    corresponding torch.device.  Raises ValueError if no match is found.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices available.")

    n = torch.cuda.device_count()

    # ── Resolve to an integer index ───────────────────────────────────────────
    if isinstance(query, int):
        idx = query
    elif isinstance(query, str) and query.startswith("cuda:"):
        try:
            idx = int(query.split(":")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Cannot parse CUDA string: {query!r}")
    else:
        # Name substring search
        needle = str(query).lower()
        matches = [i for i in range(n) if needle in torch.cuda.get_device_name(i).lower()]
        if not matches:
            available = [f"  cuda:{i}  {torch.cuda.get_device_name(i)}" for i in range(n)]
            raise ValueError(
                f"No GPU name contains {query!r}. Available devices:\n" + "\n".join(available)
            )
        if len(matches) > 1:
            names = [f"cuda:{i}  {torch.cuda.get_device_name(i)}" for i in matches]
            logger.warning(f"Multiple GPUs match {query!r}: {names}. Using first match.")
        idx = matches[0]

    if idx < 0 or idx >= n:
        raise ValueError(f"GPU index {idx} out of range (0-{n - 1}).")

    device = torch.device(f"cuda:{idx}")
    torch.cuda.set_device(idx)
    logger.info(f"Selected cuda:{idx}  {torch.cuda.get_device_name(idx)}")
    return device


def get_device(requested_device: str) -> torch.device:
    """Resolve `requested_device` ("cuda", "cuda:N", "cpu", ...) to a torch.device.

    Falls back to CPU (with a logged warning) if the requested device is
    unavailable or invalid.
    """
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                device = torch.device(requested_device)
                torch.cuda.get_device_properties(device)  # to trigger error if invalid
                logger.info(f"Using CUDA device: {device}")
                logger.debug(torch.cuda.get_device_properties(device))
                return device
            except (AssertionError, RuntimeError) as e:
                logger.warning(f"Invalid CUDA device '{requested_device}': {e}")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")

    elif requested_device == "cpu":
        logger.info("Using CPU")
        return torch.device("cpu")

    else:
        try:
            # Attempt to construct a torch.device from the string
            device = torch.device(requested_device)
            logger.info(f"Using device: {device}")
            return device
        except Exception as e:  # noqa: BLE001 -- torch.device() error types for arbitrary strings aren't enumerable; fall back to CPU
            logger.warning(f"Unrecognized device '{requested_device}': {e}. Using CPU.")

    logger.info("Using CPU")
    return torch.device("cpu")
