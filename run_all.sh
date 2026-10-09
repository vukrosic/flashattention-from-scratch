#!/bin/sh
# Runs the seven steps in order, then the tests. Python 3 with numpy.
# Step 6 ends with 131,072 tokens; the whole run took 5 minutes on a busy 16 GB laptop.
# Steps 2, 3 and 6 run the textbook version and need 3 GB of free memory.
PY="${PYTHON:-python3}"
cd "$(dirname "$0")" || exit 1
for f in step1_attention step2_table step3_rows step4_broken step5_running step6_tiled step7_traffic tests; do
  echo "--- $f"
  "$PY" "$f.py" || exit 1
done
