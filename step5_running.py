"""Step 5. A running total, and a running top.

Do not divide inside the tile. Keep the sum of the powers so far, and divide once, at the end.
The powers explode for big scores, so measure every score against the biggest one seen so far.
When a bigger one turns up, shrink what you have.
"""
import numpy as np

import attention as A

ROW = np.array([2, 1, 0, 1, 3, 1, 2, 0], np.float32)       # the row of step 4
VAL = np.array([4, 8, 6, 2, 5, 9, 1, 7], np.float32)       # one value per key, to mix


def running_total(row, val, cols=4):
    """Running total only. Fine for small scores, breaks for big ones."""
    total, mix, steps = np.float32(0), np.float32(0), []
    for j in range(0, len(row), cols):
        powers = np.exp(row[j:j + cols])
        total = total + powers.sum()
        mix = mix + powers @ val[j:j + cols]
        steps.append({'powers': powers, 'total': float(total), 'mix': float(mix)})
    return float(mix / total), steps


def running_top(row, val, cols=4):
    """Running total and running top. This is the inside of attention_tiled for one row."""
    top, total, mix, steps = np.float32(-np.inf), np.float32(0), np.float32(0), []
    for j in range(0, len(row), cols):
        new_top = max(top, row[j:j + cols].max())
        shrink = np.exp(top - new_top)
        powers = np.exp(row[j:j + cols] - new_top)
        total = total * shrink + powers.sum()
        mix = mix * shrink + powers @ val[j:j + cols]
        top = new_top
        steps.append({'top': float(top), 'shrink': float(shrink), 'powers': powers,
                      'total': float(total), 'mix': float(mix)})
    return float(mix / total), steps


def run():
    np.set_printoptions(precision=2, suppress=True)
    want = float(A.softmax(ROW[None])[0] @ VAL)
    got, steps = running_total(ROW, VAL)
    print(f'whole row at once   {want:.4f}')
    for s in steps:
        print(f'   tile powers {s["powers"]}  total {s["total"]:.2f}  mix {s["mix"]:.2f}')
    print(f'running total       {got:.4f}')
    with np.errstate(over='ignore', invalid='ignore'):
        limit = [float(np.exp(np.float32(x))) for x in (88, 89)]
        big, _ = running_total(ROW * 50, VAL)
    print(f'e to the 88 = {limit[0]:.2e}, e to the 89 = {limit[1]}')
    print(f'scores times 50, running total: {big}')
    want_big = float(A.softmax(ROW[None] * 50)[0] @ VAL)
    got_top, steps_top = running_top(ROW, VAL)
    for s in steps_top:
        print(f'   top {s["top"]:.0f}  shrink {s["shrink"]:.2f}  tile powers {s["powers"]}'
              f'  total {s["total"]:.2f}  mix {s["mix"]:.2f}')
    big_top, _ = running_top(ROW * 50, VAL)
    print(f'running top         {got_top:.4f}')
    print(f'scores times 50, running top: {big_top:.4f} (whole row at once: {want_big:.4f})')
    return {'row': ROW, 'val': VAL, 'want': want, 'got': got, 'steps': steps, 'limit': limit, 'big': big,
            'got_top': got_top, 'steps_top': steps_top, 'big_top': big_top, 'want_big': want_big}


if __name__ == '__main__':
    run()
