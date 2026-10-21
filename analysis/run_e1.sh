#!/bin/zsh
# E1 production: 3 cases x 5 seeds x {vqc 4x2, vqc 6x4, matched mlp/ff for each, mlpref}; Adam 2000 + L-BFGS 1000, n_c 512.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=3
jobs_file=logs/e1/jobs.txt; : > $jobs_file
for case in exp step smooth; do
  for seed in 0 1 2 3 4; do
    for cfg in "6 4" "4 2"; do
      set -- ${=cfg}
      for model in vqc mlp ff; do
        lr=$(python3 -c "import json;print(json.load(open('results/lrsel/selected_lrs.json'))['$case|$model|$1|$2'])")
        echo "python3 qpinn_local.py run --model $model --case $case --gamma 9 --n_q $1 --L $2 --seed $seed --iters 2000 --lbfgs 1000 --n_c 512 --lr $lr --out results/e1 > logs/e1/${model}_${case}_q$1L$2_s$seed.log 2>&1" >> $jobs_file
      done
    done
    lr=$(python3 -c "import json;print(json.load(open('results/lrsel/selected_lrs.json'))['$case|mlpref|6|4'])")
    echo "python3 qpinn_local.py run --model mlpref --case $case --gamma 9 --n_q 6 --L 4 --seed $seed --iters 2000 --lbfgs 1000 --n_c 512 --lr $lr --out results/e1 > logs/e1/mlpref_${case}_s$seed.log 2>&1" >> $jobs_file
  done
done
# VQC jobs first (slowest), 4 workers
( grep "model vqc" $jobs_file; grep -v "model vqc" $jobs_file ) | xargs -P 4 -I{} zsh -c '{}'
echo E1_DONE
