"""Step 3. The table is only a stopover, and its rows do not need each other.

So finish 64 rows, write their answers down, and reuse the same memory for the next 64.
"""
import numpy as np

import attention as A
import measure as M


def run(N=16384, rows=64):
    Q, K, V = A.tokens(N)
    want = A.attention(Q, K, V)
    got = A.attention_rows(Q, K, V, rows)
    sec = M.race({
        'table': lambda: A.attention(Q, K, V),
        'one': lambda: A.attention_one_table(Q, K, V),
        'rows': lambda: A.attention_rows(Q, K, V, rows),
    })
    out = {
        'N': N, 'rows': rows, 'sec': sec,
        'peak_table': M.peak_bytes(lambda: A.attention(Q, K, V)),
        'peak_one': M.peak_bytes(lambda: A.attention_one_table(Q, K, V)),
        'peak_rows': M.peak_bytes(lambda: A.attention_rows(Q, K, V, rows)),
        'diff': float(np.abs(got - want).max()),
        'slice': rows * N * 4, 'keys': K.nbytes, 'values': V.nbytes,
    }
    print(f'{N} tokens')
    print(f'textbook             peak {M.size(out["peak_table"]):>8}   {sec["table"]:.2f} s')
    print(f'one table, in place  peak {M.size(out["peak_one"]):>8}   {sec["one"]:.2f} s')
    print(f'{rows} rows at a time    peak {M.size(out["peak_rows"]):>8}   {sec["rows"]:.2f} s')
    print(f'biggest difference between the answers: {out["diff"]:.1e}')
    print(f'one slice of scores {M.size(out["slice"])}, all keys {M.size(out["keys"])},'
          f' all values {M.size(out["values"])}')
    return out


if __name__ == '__main__':
    run()
