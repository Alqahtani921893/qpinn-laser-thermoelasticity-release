#!/bin/zsh
# Learning-rate selection by validation (never the test grid): Adam 1000, n_c 512, seed 100, grid {1e-3, 3e-3, 1e-2, 3e-2},
# for each case, each family and configuration.  Same grid and budget for every family.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=3
jobs_file=logs/lrsel/jobs.txt; : > $jobs_file
for case in exp step smooth; do
  for lr in 1e-3 3e-3 1e-2 3e-2; do
    for cfg in "6 4" "4 2"; do set -- ${=cfg}
      for model in vqc mlp ff; do
        echo "python3 qpinn_local.py run --model $model --case $case --gamma 9 --n_q $1 --L $2 --seed 100 --iters 1000 --lbfgs 0 --n_c 512 --lr $lr --tag lr$lr --out results/lrsel > logs/lrsel/${model}_${case}_q$1L$2_lr$lr.log 2>&1" >> $jobs_file
      done
    done
    echo "python3 qpinn_local.py run --model mlpref --case $case --gamma 9 --n_q 6 --L 4 --seed 100 --iters 1000 --lbfgs 0 --n_c 512 --lr $lr --tag lr$lr --out results/lrsel > logs/lrsel/mlpref_${case}_lr$lr.log 2>&1" >> $jobs_file
  done
done
( grep "model vqc" $jobs_file; grep -v "model vqc" $jobs_file ) | xargs -P 4 -I{} zsh -c '{}'
echo LRSEL_DONE
