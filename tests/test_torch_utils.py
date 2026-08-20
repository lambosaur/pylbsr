"""Tests for torch_utils.set_seed.

Only covers set_seed -- the function newly moved here (from misc.py) and completed as part of
the ml/ml_stats restructuring; the rest of torch_utils.py (select_gpu, get_device) predates
this change and had no test coverage before it, out of scope here. Requires the `torch` extra.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pylbsr.torch_utils import set_seed


def test_set_seed_makes_python_numpy_and_torch_rngs_reproducible() -> None:
    """Calling set_seed with the same value reproduces the same draws from all three RNGs."""
    import random

    set_seed(123)
    py_draw_1 = random.random()
    np_draw_1 = np.random.rand()
    torch_draw_1 = torch.rand(1).item()

    set_seed(123)
    py_draw_2 = random.random()
    np_draw_2 = np.random.rand()
    torch_draw_2 = torch.rand(1).item()

    assert py_draw_1 == py_draw_2
    assert np_draw_1 == np_draw_2
    assert torch_draw_1 == torch_draw_2


def test_set_seed_different_seeds_diverge() -> None:
    """Different seeds produce different torch draws (sanity check against a no-op stub)."""
    set_seed(1)
    draw_a = torch.rand(1).item()

    set_seed(2)
    draw_b = torch.rand(1).item()

    assert draw_a != draw_b


def test_set_seed_configures_deterministic_cudnn_and_algorithms() -> None:
    """set_seed flips cuDNN to deterministic mode and enables deterministic algorithms."""
    set_seed(42)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled()


def test_set_seed_sets_cublas_workspace_config_env_var() -> None:
    """CUBLAS_WORKSPACE_CONFIG gets a default value, required for deterministic cuBLAS ops."""
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    set_seed(42)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
