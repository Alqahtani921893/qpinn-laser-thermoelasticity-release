#!/bin/zsh
# Ablation of the near-surface exclusion width x_min of the initial-rate terms (exp pulse, gamma 9, seed 0; otherwise the E1
# recipe: Adam 2000 + L-BFGS 1000, N_c 512, mixed sampling, E1-selected learning rates).
#   ./run_ablation_xmin.sh mlpref   -> MLP-ref (lr 0.01) at x_min in {0, 0.1, 0.3, 0.6, 1.0}   (run 21 Oct 2026, ~04:10)
#   ./run_ablation_xmin.sh circuit  -> 6q4L circuit (lr 0.1) and its matched MLP (114 params, lr 0.1) at x_min in {0, 0.1}
#                                      (run 21 Oct 2026, 07:05); their x_min = 0.3 rows are the E1 seed-0 runs (same settings).
cd "$(dirname "$0")"; export OMP_NUM_THREADS=2; mkdir -p results/ablation logs/ablation
run() { python3 qpinn_local.py run --model $1 --case exp --gamma 9 --n_q 6 --L 4 --seed 0 --iters 2000 --lbfgs 1000 --n_c 512 --lr $2 --ic_xmin $3 --tag xmin$3 --out results/ablation > logs/ablation/$1_xmin$3.log 2>&1 }
case "$1" in
  mlpref)  for x in 0.0 0.1 0.3 0.6 1.0; do run mlpref 0.01 $x; done ;;
  circuit) run vqc 0.1 0.0 & run vqc 0.1 0.1 & run mlp 0.1 0.0 & run mlp 0.1 0.1 & wait ;;
  *) echo "usage: $0 mlpref|circuit"; exit 1 ;;
esac
echo ABLATION_XMIN_DONE
