"""SP-Mem memory backend."""

try:
    from importlib.metadata import version

    __version__ = version("spmem")
except Exception:  # pragma: no cover
    __version__ = "0.1.0"

from spmem_memory.memory.main import AsyncMemory, Memory  # noqa: F401

__all__ = ["Memory", "AsyncMemory"]
