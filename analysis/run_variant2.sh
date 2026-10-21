#!/bin/zsh
# usage: run_variant2.sh NAME SAMPLING REGIME ADAM_ITERS LBFGS_ITERS W_SCALE MODEL   (exp pulse, gamma 9)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
python3 - "$2" "$3" "$4" "$5" "$6" "$7" > "logs/variants/$1.log" 2>&1 <<'PY'
import sys, torch, numpy as np; sys.path.insert(0,'.'); torch.set_num_threads(2)
import qpinn_local as q, ground_truth as gt
sampling, regime, iters, lbfgs, wsc, kind = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
m = q.make_model(kind, target=124); print('variant', sys.argv[1:], 'params', m.n_params(), flush=True)
r = q.train(m, 'exp', 9.0, regime=regime, iters=iters, lr=3e-3, n_c=1024, seed=0, log_every=500, sampling=sampling, w_scale=wsc, lbfgs_iters=lbfgs)
print({k: v for k, v in r.items() if k != 'hist'}, flush=True)
F = q.Fields(m, 9.0, 'exp')
for x, t in [(0.1,0.1),(0.1,1.0),(1.1,0.75),(2.1,1.5),(3.0,2.5)]:
    with torch.no_grad(): T, S = F.sigma(torch.tensor([x]), torch.tensor([t]))
    Te, Se = gt.exact(x, t, 'exp', 9.0); print("x=%g t=%g  T %.4f vs %.4f   sigma %.4f vs %.4f" % (x, t, T.item(), Te, S.item(), Se), flush=True)
PY
