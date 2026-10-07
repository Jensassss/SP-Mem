from __future__ import annotations

import os


DEBUG: bool = os.getenv("SPMEM_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def set_debug(enabled: bool) -> None:
    global DEBUG
    DEBUG = bool(enabled)


def debug_print(*args, **kwargs) -> None:
    if DEBUG:
        print(*args, **kwargs)

