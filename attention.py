"""Attention four ways. All four return the same answer; they differ in how much memory they need.

Q, K and V are tables with one row per token and d numbers per row (float32).
"""
import numpy as np


def softmax(S):
    """Scores of a row become shares that add up to 1."""
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return P / P.sum(axis=1, keepdims=True)


def attention(Q, K, V):
    """The textbook version. It builds the whole table."""
    S = Q @ K.T / Q.shape[1] ** 0.5        # scores: N x N
    P = softmax(S)                         # shares: N x N
    return P @ V                           # every row: a mix of the value rows


def attention_one_table(Q, K, V):
    """The textbook version written with care: one N x N table, changed in place."""
    S = Q @ K.T
    S /= Q.shape[1] ** 0.5
    S -= S.max(axis=1, keepdims=True)
    np.exp(S, out=S)
    S /= S.sum(axis=1, keepdims=True)
    return S @ V


def attention_rows(Q, K, V, rows=64):
    """A few rows of the table at a time."""
    O = np.empty_like(Q)
    for i in range(0, len(Q), rows):
        O[i:i + rows] = attention_one_table(Q[i:i + rows], K, V)
    return O


def attention_tiled(Q, K, V, rows=512, cols=512):
    """One tile of scores at a time. No table is ever stored."""
    N, d = Q.shape
    O = np.empty_like(Q)
    for i in range(0, N, rows):
        q = Q[i:i + rows] / d ** 0.5                         # a block of queries
        top = np.full((len(q), 1), -np.inf, Q.dtype)         # biggest score so far
        total = np.zeros((len(q), 1), Q.dtype)               # running total of the row
        mix = np.zeros((len(q), d), Q.dtype)                 # running mix of values
        for j in range(0, N, cols):
            tile = q @ K[j:j + cols].T                       # scores: rows x cols
            new_top = np.maximum(top, tile.max(axis=1, keepdims=True))
            shrink = np.exp(top - new_top)                   # 1 if the top did not move
            np.exp(tile - new_top, out=tile)
            total = total * shrink + tile.sum(axis=1, keepdims=True)
            mix = mix * shrink + tile @ V[j:j + cols]
            top = new_top
        O[i:i + rows] = mix / total                          # divide once, at the end
    return O


def tokens(N, d=64, seed=0):
    """Three random tables Q, K, V. In a real model they come from the layer before."""
    rng = np.random.default_rng(seed)
    return tuple(rng.standard_normal((N, d), dtype=np.float32) for _ in range(3))
