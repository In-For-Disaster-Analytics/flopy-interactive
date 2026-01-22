"""Performance helpers for optional profiling."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

PERF_ENABLED = os.environ.get("FLOPY_PROFILE", "").strip().lower() in ("1", "true", "yes", "on")
try:
    PERF_MIN_MS = float(os.environ.get("FLOPY_PROFILE_MIN_MS", "0"))
except ValueError:
    PERF_MIN_MS = 0.0


def perf_note(message: str) -> None:
    """Emit a profiling note when profiling is enabled."""
    if not PERF_ENABLED:
        return
    print(f"[perf] {message}", file=sys.stderr)


@contextmanager
def perf_timer(label: str):
    """Time a block and emit a formatted line when profiling is enabled."""
    if not PERF_ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms >= PERF_MIN_MS:
            print(f"[perf] {label}: {elapsed_ms:.1f}ms", file=sys.stderr)


def perf_call(label: str, func, *args, **kwargs):
    """Time a function call and return its output."""
    if not PERF_ENABLED:
        return func(*args, **kwargs)
    start = time.perf_counter()
    try:
        return func(*args, **kwargs)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms >= PERF_MIN_MS:
            print(f"[perf] {label}: {elapsed_ms:.1f}ms", file=sys.stderr)
