import logging

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "torch is required for pylbsr.torch_utils. Install with `pip install pylbsr[ml]`."
    ) from e

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())



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

    Examples
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
        raise ValueError(f"GPU index {idx} out of range (0–{n - 1}).")

    device = torch.device(f"cuda:{idx}")
    torch.cuda.set_device(idx)
    logger.info(f"Selected cuda:{idx}  {torch.cuda.get_device_name(idx)}")
    return device


def get_device(requested_device: str) -> torch.device:
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
        except Exception as e:
            logger.warning(f"Unrecognized device '{requested_device}': {e}. Using CPU.")

    logger.info("Using CPU")
    return torch.device("cpu")

