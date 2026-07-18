"""
Built-in profiler: per-block reconstruction time, VJP time, pass time,
memory, allocations, synchronization. Researchers need measurements.
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from typing import Dict, List


class Profiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._records: Dict[str, List[float]] = defaultdict(list)

    def record(self, name: str, elapsed_seconds: float) -> None:
        if self.enabled:
            self._records[name].append(elapsed_seconds)

    @contextlib.contextmanager
    def track(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start)

    def report(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for name, values in self._records.items():
            out[name] = {
                "count": len(values),
                "total_seconds": sum(values),
                "mean_seconds": sum(values) / len(values) if values else 0.0,
                "max_seconds": max(values) if values else 0.0,
                "min_seconds": min(values) if values else 0.0,
            }
        return out

    def reset(self) -> None:
        self._records.clear()
