# FlashAttention from scratch

Code for the video **The Table That Never Gets Built**.
Attention compares every token with every token, so the textbook version builds a table with one cell per pair. At 16,384 tokens that table is 1 GB, and the textbook version holds three of them.
These seven small programs remove the table step by step: rows, then tiles, then a running total, then a running top. The last version gives the same answer with 6.3 MB instead of 3 GB. About 580 lines of Python 3 with numpy.

```bash
pip install numpy
./run_all.sh          # runs all seven steps and the tests; expected output is in demo-output.txt
```

Steps 2, 3 and 6 run the textbook version on 16,384 tokens and need 3 GB of free memory.
Step 6 ends with 131,072 tokens. The whole run took 5 minutes on a 16 GB laptop that was busy with other work.
The seconds in `demo-output.txt` come from that busy machine and change from run to run. Memory, traffic and the differences between the answers do not.

| File | Idea | What the run shows |
| --- | --- | --- |
| `attention.py` | The four versions | `attention` (textbook), `attention_one_table`, `attention_rows`, `attention_tiled`. All four return the same answer. |
| `measure.py` | The two rulers | Peak extra memory of a call, and its best time. |
| `step1_attention.py` | What attention computes | Six tokens: scores, shares that add up to 1 in every row, the answer. |
| `step2_table.py` | The table grows with the square | 1,024 tokens 4 MB, 16,384 tokens 1 GB, 131,072 tokens 64 GB. The textbook version peaks at three tables: 3 GB. |
| `step3_rows.py` | Rows stand alone | 64 rows at a time: peak 8.0 MB instead of 3 GB, difference to the textbook answer 0. |
| `step4_broken.py` | Tiles break the shares | A row of eight scores cut into two tiles adds up to 2, not 1. At 4,096 tokens the answer is off by 0.93. |
| `step5_running.py` | Divide at the very end | A running total and a running mix give 4.4875, the same as the whole row. With scores 50 times larger the powers overflow to `inf` and the answer is `nan`; a running top fixes that. |
| `step6_tiled.py` | The whole function | 16,384 tokens: peak 6.3 MB, difference 4.0e-07 (rounding of 32-bit numbers). 131,072 tokens: peak 2.3 MB besides the answer, 256 rows checked the slow way. |
| `step7_traffic.py` | Why the real one is faster | Numbers moved between a 192 KB fast memory and the big memory: 1,077,936,128 with the table, 270,532,608 with tiles of 128 × 128. That is 4.0 times less. |
| `tests.py` | Checks of the parts | 6 tests, among them: with 64-bit numbers the four versions agree to 13 digits. |

## What this is not

This is the idea of FlashAttention (Dao, Fu, Ermon, Rudra, Ré, 2022, arXiv 2205.14135), not the program.
The real one is a single program that runs on the graphics card and keeps each tile in the card's small fast memory. That is where its speed comes from: 41.7 ms down to 7.3 ms for GPT-2 medium in the paper.
On a laptop with numpy the arithmetic is the slow part, so this version saves memory and almost no time.
Step 7 counts traffic with a simple model: every number read from or written to the big memory counts once.
To use the real thing, call `torch.nn.functional.scaled_dot_product_attention` in PyTorch.

## Try it

A language model may only look back, never ahead. Teach `attention_tiled` that rule and skip every tile that lies completely in the forbidden half.
`python3 exercise.py` shows one solution: at 8,192 tokens with tiles of 512, 136 of 256 tiles are enough.

MIT licence.
