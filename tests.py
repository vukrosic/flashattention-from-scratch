"""Checks of the parts. Run: python3 tests.py"""
import numpy as np

import attention as A
import exercise as E
import step5_running as S5
import step7_traffic as S7


def test_shares_add_up_to_one():
    S = np.random.default_rng(0).standard_normal((50, 70)).astype(np.float32) * 5
    P = A.softmax(S)
    assert np.allclose(P.sum(axis=1), 1, atol=1e-5) and (P >= 0).all()


def test_four_versions_agree():
    for N in (1, 7, 200, 777):                                # sizes that do not divide by the tile
        Q, K, V = A.tokens(N, d=16, seed=N)
        want = A.attention(Q, K, V)
        assert np.abs(A.attention_one_table(Q, K, V) - want).max() < 1e-5
        assert np.abs(A.attention_rows(Q, K, V, 64) - want).max() < 1e-5
        for rows, cols in ((1, 1), (3, 5), (64, 32), (1000, 1000)):
            assert np.abs(A.attention_tiled(Q, K, V, rows, cols) - want).max() < 1e-5


def test_exact_with_sixteen_digit_numbers():
    Q, K, V = (x.astype(np.float64) for x in A.tokens(600, d=32, seed=1))
    assert np.abs(A.attention_tiled(Q, K, V, 64, 64) - A.attention(Q, K, V)).max() < 1e-13


def test_big_scores_do_not_explode():
    Q, K, V = A.tokens(300, d=16, seed=2)
    O = A.attention_tiled(Q * 40, K, V, 32, 32)               # scores far above 89
    assert np.isfinite(O).all() and np.abs(O - A.attention(Q * 40, K, V)).max() < 1e-5
    assert np.isnan(S5.running_total(S5.ROW * 50, S5.VAL)[0])  # without the running top it does explode


def test_traffic_count():
    N, d = 1024, 64
    assert S7.traffic_table(N, d) == 4 * N * N + 4 * N * d
    assert S7.traffic_tiled(N, d, 128, 128) == 2 * N * d + (N // 128) * 2 * N * d
    assert S7.traffic_tiled(N, d, 100, 60) == 2 * N * d + 11 * 2 * N * d      # 11 blocks of 100 rows


def test_look_back_only():
    Q, K, V = A.tokens(333, d=16, seed=4)
    want = E.attention_look_back(Q, K, V)
    assert np.abs(E.attention_tiled_look_back(Q, K, V, 50, 40) - want).max() < 1e-5
    assert np.allclose(want[0], V[0], atol=1e-6)              # the first token can only look at itself


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    with np.errstate(over='ignore', invalid='ignore'):
        for t in tests:
            t()
            print('ok', t.__name__)
    print(f'{len(tests)} tests passed')
