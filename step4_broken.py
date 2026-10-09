"""Step 4. Cut the keys into blocks too, and softmax breaks.

A share is a score's part of its whole row. A tile only sees a piece of the row.
If every tile makes its own shares, every tile adds up to 1, and the row adds up to the number of tiles.
"""
import numpy as np

import attention as A

ROW = np.array([2, 1, 0, 1, 3, 1, 2, 0], np.float32)       # one row of scores, cut into two tiles of four


def attention_tiled_wrong(Q, K, V, cols=512):
    """Softmax inside each tile on its own. Wrong on purpose."""
    O = np.zeros_like(Q)
    for j in range(0, len(K), cols):
        tile = Q @ K[j:j + cols].T / Q.shape[1] ** 0.5
        O += A.softmax(tile) @ V[j:j + cols]
    return O


def run(N=4096, cols=512):
    np.set_printoptions(precision=2, suppress=True)
    right = A.softmax(ROW[None])[0]
    wrong = np.concatenate([A.softmax(ROW[None, :4])[0], A.softmax(ROW[None, 4:])[0]])
    print('scores           ', ROW)
    print('shares, whole row', right, 'sum', round(float(right.sum()), 2))
    print('shares, per tile ', wrong, 'sum', round(float(wrong.sum()), 2))
    Q, K, V = A.tokens(N)
    want = A.attention(Q, K, V)
    got = attention_tiled_wrong(Q, K, V, cols)
    diff = float(np.abs(got - want).max())
    print(f'{N} tokens in {N // cols} tiles per row: biggest difference to the right answer {diff:.2f}')
    return {'row': ROW, 'right': right, 'wrong': wrong, 'N': N, 'tiles': N // cols, 'diff': diff,
            'typical': float(np.abs(want).mean())}


if __name__ == '__main__':
    run()
