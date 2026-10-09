"""Step 6. The whole thing: tiles in both directions, a running top and a running total.

Same answer as the textbook version, and no table. Then a size the textbook version cannot run here.
"""
import time
import tracemalloc

import numpy as np

import attention as A
import measure as M


def run(N=16384, tile=512, big=131072, sweep=(64, 128, 256, 512, 1024, 2048, 4096)):
    Q, K, V = A.tokens(N)
    want = A.attention(Q, K, V)
    got = A.attention_tiled(Q, K, V, tile, tile)
    sec = M.race({
        'table': lambda: A.attention(Q, K, V),
        'one': lambda: A.attention_one_table(Q, K, V),
        'tiled': lambda: A.attention_tiled(Q, K, V, tile, tile),
    })
    out = {
        'N': N, 'tile': tile, 'sec': sec, 'diff': float(np.abs(got - want).max()),
        'typical': float(np.abs(want).mean()),
        'peak_table': M.peak_bytes(lambda: A.attention(Q, K, V)),
        'peak_tiled': M.peak_bytes(lambda: A.attention_tiled(Q, K, V, tile, tile)),
    }
    print(f'{N} tokens, tiles of {tile} x {tile}')
    print(f'textbook             peak {M.size(out["peak_table"]):>8}   {sec["table"]:.2f} s')
    print(f'one table, in place  peak {M.size(N * N * 4):>8}   {sec["one"]:.2f} s')
    print(f'tiled                peak {M.size(out["peak_tiled"]):>8}   {sec["tiled"]:.2f} s')
    print(f'biggest difference between the answers: {out["diff"]:.1e}')
    print(f'a typical number in the answer: {out["typical"]:.3f}')
    times = M.race({t: (lambda t=t: A.attention_tiled(Q, K, V, t, t)) for t in sweep}, rounds=3)
    out['sweep'] = [{'tile': t, 'bytes': t * t * 4, 'sec': times[t]} for t in sweep]
    for s in out['sweep']:
        print(f'   tile {s["tile"]:>5} x {s["tile"]:<5} {M.size(s["bytes"]):>9}   {s["sec"]:.2f} s')
    del want, got
    Q, K, V = A.tokens(big)
    tracemalloc.start()
    t = time.perf_counter()
    O = A.attention_tiled(Q, K, V, tile, tile)
    big_sec = time.perf_counter() - t
    big_peak = tracemalloc.get_traced_memory()[1] - O.nbytes
    tracemalloc.stop()
    pick = np.arange(0, big, big // 256)                       # check 256 rows the slow, sure way
    big_diff = float(np.abs(A.attention_rows(Q[pick], K, V, 32) - O[pick]).max())
    out.update(big=big, big_sec=big_sec, big_peak=big_peak, big_table=big * big * 4, big_diff=big_diff)
    print(f'{big} tokens: one table would be {M.size(big * big * 4)}')
    print(f'tiled: peak {M.size(big_peak)} besides the answer')
    print(f'256 rows checked the slow way, biggest difference {big_diff:.1e}')
    return out


if __name__ == '__main__':
    run()
