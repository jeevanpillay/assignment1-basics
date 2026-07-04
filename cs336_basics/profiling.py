# profiling.py
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext


class NullProfiler:
    def section(self, name: str):
        return nullcontext()

    def report(self):
        pass


class SectionProfiler:
    def __init__(self):
        self.times = defaultdict(float)
        self.calls = defaultdict(int)

    @contextmanager
    def section(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter() - start
            self.calls[name] += 1

    def report(self):
        for name, total in sorted(self.times.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:24} {total:10.4f}s calls={self.calls[name]}")
