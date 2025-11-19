
try:
    from ._version import __version__
except ImportError:
    import importlib.metadata
    __version__ = importlib.metadata.version(__name__)


from . import misc, notebooks, plotting, torch_utils

__all__ = [
    "misc",
    "notebooks",
    "plotting",
    "torch_utils",
]