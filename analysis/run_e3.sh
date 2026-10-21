#!/bin/zsh
# Learning rates: the E1-selected exponential-pulse rates (0.1 for every VQC, MLP-matched and FF-matched cell; MLP-ref 0.01);
# the data-only regime gets its own validation-based selection (vqc 6q4L and matched mlp, N_d 256, Adam 1000, same grid).
# E3: (i) error vs parameter count (Pareto) on exp/gamma 9, 2 seeds: VQC configs and matched MLP/FF for each;
#     (ii) error vs collocation count N_c in {64,256,1024,4096} (physics-only) for VQC 6x4 and matched MLP, 2 seeds;
#     (iii) error vs labelled points N_d in {16,64,256,1024} (data-only, no PDE loss), same models, 2 seeds.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=3
mkdir -p results/e3 logs/e3 results/e3_lrsel_data; jobs_file=logs/e3/jobs_lrsel_data.txt; : > $jobs_file
for lr in 1e-3 3e-3 1e-2 3e-2 1e-1; do for model in vqc mlp; do
  echo "python3 qpinn_local.py run --model $model --case exp --gamma 9 --n_q 6 --L 4 --seed 100 --iters 1000 --lbfgs 0 --n_d 256 --regime data --lr $lr --tag datalr$lr --out results/e3_lrsel_data > logs/e3/lrsel_data_${model}_lr$lr.log 2>&1" >> $jobs_file
done; done
xargs -P 3 -I{} zsh -c '{}' < $jobs_file
python3 - <<'PY'
import json, glob, collections
rows = collections.defaultdict(dict)
for f in glob.glob('results/e3_lrsel_data/*.json'):
    r = json.load(open(f)); rows[r['model']][r['lr']] = r['val']
sel = {m: min(v, key=v.get) for m, v in rows.items()}; json.dump(sel, open('results/e3_lrsel_data/selected_lrs_data.json', 'w')); print('data-only lrs', sel)
PY
LR_VQC_DATA=$(python3 -c "import json;print(json.load(open('results/e3_lrsel_data/selected_lrs_data.json'))['vqc'])")
LR_MLP_DATA=$(python3 -c "import json;print(json.load(open('results/e3_lrsel_data/selected_lrs_data.json'))['mlp'])")
jobs_file=logs/e3/jobs.txt; : > $jobs_file
for seed in 0 1; do
  for cfg in "4 2" "4 4" "6 2" "6 4" "6 8" "8 2"; do set -- ${=cfg}
    for model in vqc mlp ff; do
      echo "python3 qpinn_local.py run --model $model --case exp --gamma 9 --n_q $1 --L $2 --seed $seed --iters 2000 --lbfgs 1000 --n_c 512 --lr 0.1 --tag pareto --out results/e3 > logs/e3/pareto_${model}_q$1L$2_s$seed.log 2>&1" >> $jobs_file
    done
  done
  for nc in 64 256 1024 4096; do
    for model in vqc mlp; do
      echo "python3 qpinn_local.py run --model $model --case exp --gamma 9 --n_q 6 --L 4 --seed $seed --iters 2000 --lbfgs 1000 --n_c $nc --lr 0.1 --tag nc$nc --out results/e3 > logs/e3/nc${nc}_${model}_s$seed.log 2>&1" >> $jobs_file
    done
  done
  for nd in 16 64 256 1024; do
    for model in vqc mlp; do
      echo "python3 qpinn_local.py run --model $model --case exp --gamma 9 --n_q 6 --L 4 --seed $seed --iters 2000 --lbfgs 0 --n_d $nd --regime data --lr $([ $model = vqc ] && echo $LR_VQC_DATA || echo $LR_MLP_DATA) --tag nd$nd --out results/e3 > logs/e3/nd${nd}_${model}_s$seed.log 2>&1" >> $jobs_file
    done
  done
done
( grep "model vqc" $jobs_file; grep -v "model vqc" $jobs_file ) | xargs -P ${E3_WORKERS:-3} -I{} zsh -c '{}'
echo E3_DONE
