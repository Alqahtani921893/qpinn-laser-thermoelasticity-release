#!/bin/zsh
# extra learning rate 1e-1 for every family (grid extension, same budget as run_lrsel.sh)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=3
jobs_file=logs/lrsel/jobs_hi.txt; : > $jobs_file
for case in exp step smooth; do
  for cfg in "6 4" "4 2"; do set -- ${=cfg}
    for model in vqc mlp ff; do
      echo "python3 qpinn_local.py run --model $model --case $case --gamma 9 --n_q $1 --L $2 --seed 100 --iters 1000 --lbfgs 0 --n_c 512 --lr 1e-1 --tag lr1e-1 --out results/lrsel > logs/lrsel/${model}_${case}_q$1L$2_lr1e-1.log 2>&1" >> $jobs_file
    done
  done
  echo "python3 qpinn_local.py run --model mlpref --case $case --gamma 9 --n_q 6 --L 4 --seed 100 --iters 1000 --lbfgs 0 --n_c 512 --lr 1e-1 --tag lr1e-1 --out results/lrsel > logs/lrsel/mlpref_${case}_lr1e-1.log 2>&1" >> $jobs_file
done
( grep "model vqc" $jobs_file; grep -v "model vqc" $jobs_file ) | xargs -P 2 -I{} zsh -c '{}'
echo LRSEL_HI_DONE
