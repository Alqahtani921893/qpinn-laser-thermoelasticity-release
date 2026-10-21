#!/bin/zsh
# usage: run_test.sh NAME MODEL REGIME ADAM_ITERS LBFGS_ITERS [N_Q L]   (exp pulse, gamma 9, mixed sampling, direct (T,sigma))
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
python3 - "$2" "$3" "$4" "$5" "${6:-6}" "${7:-4}" > "logs/variants/$1.log" 2>&1 <<'PY'
import sys, torch, numpy as np, json; sys.path.insert(0,'.'); torch.set_num_threads(2)
import qpinn_local as q, ground_truth as gt
kind, regime, iters, lbfgs, n_q, L = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
target = q.VQC(n_q, L).n_params()
m = q.make_model(kind, n_q, L, target=target); print('test', sys.argv[1:], 'params', m.n_params(), flush=True)
r = q.train(m, 'exp', 9.0, regime=regime, iters=iters, lr=3e-3 if kind != 'vqc' else 5e-3, n_c=1024 if kind != 'vqc' else 512, seed=0, log_every=250, sampling='mixed', lbfgs_iters=lbfgs)
print({k: v for k, v in r.items() if k != 'hist'}, flush=True)
json.dump(r, open('results/test_%s.json' % sys.argv[1].replace(' ', '_'), 'w'), indent=1)
PY
