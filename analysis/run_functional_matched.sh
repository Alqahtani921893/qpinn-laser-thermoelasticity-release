#!/bin/zsh
# Classical nets matched to the FUNCTIONAL circuit parameter counts (30 for 4q2L, 88 for 6q4L):
# learning-rate selection (Adam 1000, seed 100, 5-value grid) then 5 seeds with the selected rate, Adam 2000 + L-BFGS 1000.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
mkdir -p results/fmatch_lrsel results/fmatch logs/fmatch
jobs=logs/fmatch/jobs_lrsel.txt; : > $jobs
for case in exp step smooth; do for cfg in "6 4 88" "4 2 30"; do set -- ${=cfg}; for model in mlp ff; do for lr in 1e-3 3e-3 1e-2 3e-2 1e-1; do
  echo "python3 qpinn_local.py run --model $model --case $case --gamma 9 --n_q $1 --L $2 --target $3 --seed 100 --iters 1000 --lbfgs 0 --n_c 512 --lr $lr --tag f$3_lr$lr --out results/fmatch_lrsel > logs/fmatch/lrsel_${model}_${case}_q$1L$2_lr$lr.log 2>&1" >> $jobs
done; done; done; done
xargs -P 3 -I{} zsh -c '{}' < $jobs
python3 - <<'PY'
import json, glob, collections
rows = collections.defaultdict(dict)
for f in glob.glob('results/fmatch_lrsel/*.json'):
    r = json.load(open(f)); rows[(r['case'], r['model'], r['n_q'], r['L'])][r['lr']] = r['val']
sel = {('%s|%s|%d|%d' % k): min(v, key=v.get) for k, v in rows.items()}
json.dump(sel, open('results/fmatch_lrsel/selected_lrs.json', 'w'), indent=1); print(sel)
PY
jobs=logs/fmatch/jobs_runs.txt; : > $jobs
for case in exp step smooth; do for cfg in "6 4 88" "4 2 30"; do set -- ${=cfg}; for model in mlp ff; do
  lr=$(python3 -c "import json;print(json.load(open('results/fmatch_lrsel/selected_lrs.json'))['$case|$model|$1|$2'])")
  for seed in 0 1 2 3 4; do
    echo "python3 qpinn_local.py run --model $model --case $case --gamma 9 --n_q $1 --L $2 --target $3 --seed $seed --iters 2000 --lbfgs 1000 --n_c 512 --lr $lr --tag f$3 --out results/fmatch > logs/fmatch/run_${model}_${case}_q$1L$2_s$seed.log 2>&1" >> $jobs
  done
done; done; done
xargs -P 3 -I{} zsh -c '{}' < $jobs
echo FMATCH_DONE
