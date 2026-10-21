#!/bin/zsh
# detached chain: wait for E1, then E4 (noise), then E3 (efficiency sweeps)
cd "$(dirname "$0")"
until grep -q E1_DONE logs/e1/run_e1.out 2>/dev/null; do sleep 120; done
echo "E1 finished at $(date)"
OMP_NUM_THREADS=4 python3 run_e4.py > logs/e4.log 2>&1
echo "E4 finished at $(date): $(tail -1 logs/e4.log)"
E3_WORKERS=3 ./run_e3.sh > logs/e3_run.out 2>&1
echo "E3 finished at $(date)"
echo CHAIN_DONE
