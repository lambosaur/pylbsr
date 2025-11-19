import logging

import numpy as np
import torch

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

