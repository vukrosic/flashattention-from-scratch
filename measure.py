"""Two rulers: peak extra memory of a call, and its best time.

On a busy machine, run with TIMES=0: the stopwatch is skipped and every time prints as nan.
"""
import os
import time
import tracemalloc

TIMES = os.environ.get('TIMES', '1') != '0'

GB = 1024 ** 3      # one gigabyte = 1,073,741,824 bytes, the way a laptop counts its memory
MB = 1024 ** 2


def peak_bytes(fn):
    """The most extra memory held at any moment while fn runs."""
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def seconds(fn, repeat=3):
    """Best time of a few runs, so that a busy moment on the machine does not count."""
    if not TIMES:
        return float('nan')
    best = float('inf')
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def race(runners, rounds=5):
    """Best time of each runner. They take turns, so a busy moment hits all of them alike."""
    if not TIMES:
        return {name: float('nan') for name in runners}
    best = {name: float('inf') for name in runners}
    for _ in range(rounds):
        for name, fn in runners.items():
            t = time.perf_counter()
            fn()
            best[name] = min(best[name], time.perf_counter() - t)
    return best


def size(n_bytes):
    return f'{n_bytes / GB:.2f} GB' if n_bytes >= GB / 2 else f'{n_bytes / MB:.1f} MB'
