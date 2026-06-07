from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass


@dataclass
class ProfileResult:
    elapsed_sec: float
    python_peak_mb: float


class TimerMemory:
    """Measure wall-clock time and approximate memory usage.

    Notes:
    - `python_peak_mb` comes from tracemalloc and captures Python-level allocations.
    """

    def __enter__(self):
        tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._elapsed = time.perf_counter() - self._t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.python_peak_mb = peak / (1024 * 1024)
        self.result = ProfileResult(
            elapsed_sec=self._elapsed,
            python_peak_mb=self.python_peak_mb,
        )
        return False
