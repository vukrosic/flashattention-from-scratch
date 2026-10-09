"""Exercise. A language model may only look back.

Token 5 may use the keys of tokens 1 to 5, never of token 6 or later: a score for a later key counts
as minus infinity. A tile that lies completely above that diagonal does not have to be computed at all.
Add this to attention_tiled. Check it against the textbook version, and time how much the skipped
tiles save. Try it yourself first, then run this file.
"""
import numpy as np

import attention as A
import measure as M


def attention_look_back(Q, K, V):
    """Textbook version with the rule: the whole table, later keys set to minus infinity."""
    S = Q @ K.T / Q.shape[1] ** 0.5
    S[np.triu_indices(len(Q), 1)] = -np.inf
    return A.softmax(S) @ V


def attention_tiled_look_back(Q, K, V, rows=512, cols=512):
    N, d = Q.shape
    O = np.empty_like(Q)
    for i in range(0, N, rows):
        q = Q[i:i + rows] / d ** 0.5
        me = np.arange(i, i + len(q))[:, None]               # the position of every row in this block
        top = np.full((len(q), 1), -np.inf, Q.dtype)
        total = np.zeros((len(q), 1), Q.dtype)
        mix = np.zeros((len(q), d), Q.dtype)
        for j in range(0, i + len(q), cols):                 # tiles to the right of the block are skipped
            tile = q @ K[j:j + cols].T
            tile[np.arange(j, j + tile.shape[1])[None] > me] = -np.inf
            new_top = np.maximum(top, tile.max(axis=1, keepdims=True))
            shrink = np.exp(top - new_top)
            np.exp(tile - new_top, out=tile)
            total = total * shrink + tile.sum(axis=1, keepdims=True)
            mix = mix * shrink + tile @ V[j:j + cols]
            top = new_top
        O[i:i + rows] = mix / total
    return O


def run(N=8192, tile=512):
    Q, K, V = A.tokens(N)
    diff = float(np.abs(attention_tiled_look_back(Q, K, V, tile, tile) - attention_look_back(Q, K, V)).max())
    sec = M.race({
        'all': lambda: A.attention_tiled(Q, K, V, tile, tile),
        'back': lambda: attention_tiled_look_back(Q, K, V, tile, tile),
    })
    blocks = -(-N // tile)
    done = sum(-(-(i * tile + min(tile, N - i * tile)) // tile) for i in range(blocks))
    print(f'{N} tokens: biggest difference to the textbook version with the rule {diff:.1e}')
    print(f'tiles computed: {done} of {blocks * blocks}')
    print(f'all tiles {sec["all"]:.2f} s, look back only {sec["back"]:.2f} s')
    return {'N': N, 'diff': diff, 'done': done, 'all': blocks * blocks, 'sec': sec}


if __name__ == '__main__':
    run()
