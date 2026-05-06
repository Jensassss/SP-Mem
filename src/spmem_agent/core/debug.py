from __future__ import annotations

DEBUG: bool = True


def set_debug(enabled: bool) -> None:
    global DEBUG
    DEBUG = bool(enabled)


def debug_print(*args, **kwargs) -> None:
    if DEBUG:
        print(*args, **kwargs)

