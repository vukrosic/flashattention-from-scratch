"""Step 2. The table grows with the square of the number of tokens.

Twice the tokens: four times the table. We measure the memory and the time of the textbook version.
"""
import attention as A
import measure as M

LAPTOP = 16 * M.GB


def run(sizes=(1024, 2048, 4096, 8192, 16384), ahead=(32768, 65536, 131072)):
    rows = []
    print(' tokens   one table   measured peak   seconds')
    for N in sizes:
        Q, K, V = A.tokens(N)
        peak = M.peak_bytes(lambda: A.attention(Q, K, V))
        sec = M.seconds(lambda: A.attention(Q, K, V))
        rows.append({'N': N, 'table': N * N * 4, 'peak': peak, 'sec': sec})
        print(f'{N:>7}   {M.size(N * N * 4):>9}   {M.size(peak):>13}   {sec:7.3f}')
    per_table = rows[-1]['peak'] / rows[-1]['table']
    print(f'the textbook version holds {per_table:.1f} tables at its peak')
    future = []
    for N in ahead:                      # not run: only multiplied out
        need = N * N * 4
        future.append({'N': N, 'table': need})
        print(f'{N:>7}   {M.size(need):>9}   (one table; this laptop has {M.size(LAPTOP)})')
    return {'rows': rows, 'future': future, 'per_table': per_table, 'laptop': LAPTOP}


if __name__ == '__main__':
    run()
