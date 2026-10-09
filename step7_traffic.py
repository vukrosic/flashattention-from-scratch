"""Step 7. Why the real one is faster: count the traffic.

A graphics card computes in a small fast memory and keeps its data in a big slow one.
Here we count every number that has to cross between the two, for the table and for the tiles.
Then we time the parts of the table version on this machine, to see what it is really waiting for.
"""
import time

import numpy as np

import attention as A
import measure as M

FAST = 192 * 1024       # bytes of fast memory beside one group of cores on an Nvidia A100 (paper, section 2.1)


def traffic_table(N, d):
    """Numbers that cross when the table is built in the big memory (Algorithm 0 of the paper)."""
    moved = 2 * N * d            # fetch all queries and all keys
    moved += N * N               # store the table of scores
    moved += N * N               # fetch it again for softmax
    moved += N * N               # store the table of shares
    moved += N * N + N * d       # fetch the shares and all values
    moved += N * d               # store the answer
    return moved


def traffic_tiled(N, d, rows, cols):
    """Numbers that cross when only blocks travel and every tile lives and dies in the fast memory."""
    moved = 0
    for i in range(0, N, rows):
        r = min(rows, N - i)
        moved += r * d                       # fetch a block of queries
        for j in range(0, N, cols):
            c = min(cols, N - j)
            moved += 2 * c * d               # fetch a block of keys and a block of values
        moved += r * d                       # store the finished rows of the answer
    return moved


def split(N=16384, rounds=3):
    """Best time of each part of the one-table version."""
    Q, K, V = A.tokens(N)
    best = {'scores': 9e9, 'power': 9e9, 'mix': 9e9, 'rest': 9e9}
    for _ in range(rounds):
        t0 = time.perf_counter()
        S = Q @ K.T
        t1 = time.perf_counter()
        S /= Q.shape[1] ** 0.5
        S -= S.max(axis=1, keepdims=True)
        t2 = time.perf_counter()
        np.exp(S, out=S)
        t3 = time.perf_counter()
        S /= S.sum(axis=1, keepdims=True)
        t4 = time.perf_counter()
        O = S @ V
        t5 = time.perf_counter()
        del S, O
        for k, v in (('scores', t1 - t0), ('power', t3 - t2), ('mix', t5 - t4), ('rest', t2 - t1 + t4 - t3)):
            best[k] = min(best[k], v)
    return best


def run(N=16384, d=64, tile=128):
    table = traffic_table(N, d)
    tiled = traffic_tiled(N, d, tile, tile)
    print(f'{N} tokens, {d} numbers per token, tiles of {tile} x {tile}')
    print(f'one tile of scores: {tile * tile * 4 // 1024} KB. Fast memory: {FAST // 1024} KB.'
          f' One table: {M.size(N * N * 4)}.')
    print(f'table version moves {table:>13,} numbers ({M.size(table * 4)})')
    print(f'tiled version moves {tiled:>13,} numbers ({M.size(tiled * 4)})')
    print(f'{table / tiled:.1f} times less traffic')
    parts = split(N) if M.TIMES else dict.fromkeys(('scores', 'power', 'mix', 'rest'), float('nan'))
    whole = sum(parts.values())
    for k, v in parts.items():
        print(f'   {k:<7} {v:.3f} s  {100 * v / whole:3.0f} %')
    return {'N': N, 'd': d, 'tile': tile, 'fast': FAST, 'table': table, 'tiled': tiled,
            'ratio': table / tiled, 'parts': parts, 'whole': whole}


if __name__ == '__main__':
    run()
