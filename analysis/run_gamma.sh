#!/bin/zsh
# Secondary wave speeds gamma in {25, 49}: VQC 6x4, matched MLP and FF, MLP-ref, exp pulse, 3 seeds; learning rates = the E1-selected exp-pulse rates (0.1; MLP-ref 0.01).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=3
mkdir -p results/gamma logs/gamma; jobs_file=logs/gamma/jobs.txt; : > $jobs_file
for g in 25 49; do for seed in 0 1 2; do for model in vqc mlp ff mlpref; do
  echo "python3 qpinn_local.py run --model $model --case exp --gamma $g --n_q 6 --L 4 --seed $seed --iters 2000 --lbfgs 1000 --n_c 512 --lr $([ $model = mlpref ] && echo 0.01 || echo 0.1) --out results/gamma > logs/gamma/${model}_g${g}_s$seed.log 2>&1" >> $jobs_file
done; done; done
( grep "model vqc" $jobs_file; grep -v "model vqc" $jobs_file ) | xargs -P ${G_WORKERS:-3} -I{} zsh -c '{}'
echo GAMMA_DONE
