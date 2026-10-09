"""Step 1. What attention computes, on six tokens.

Every token asks a question (its row of Q). Every token offers a key (its row of K) and a value (V).
A score says how well a question fits a key. Softmax turns a row of scores into shares.
The answer for a token is the mix of all values, weighted by its shares.
"""
import numpy as np

import attention as A

WORDS = ['the', 'cat', 'sat', 'on', 'the', 'mat']


def run():
    Q, K, V = A.tokens(len(WORDS), d=4, seed=3)
    S = Q @ K.T / Q.shape[1] ** 0.5
    P = A.softmax(S)
    O = P @ V
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print('scores, one row per token (6 x 6)')
    print(S)
    print('shares, every row adds up to 1')
    print(P)
    print('row sums', P.sum(axis=1))
    print('answer, one row per token (6 x 4)')
    print(O)
    assert np.allclose(O, A.attention(Q, K, V))
    return {'words': WORDS, 'S': S, 'P': P, 'O': O, 'V': V}


if __name__ == '__main__':
    run()
