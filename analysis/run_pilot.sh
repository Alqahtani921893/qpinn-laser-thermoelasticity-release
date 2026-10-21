#!/bin/zsh
# E1 pilot: exp pulse, gamma=9, VQC 6q/4L vs matched MLP vs matched FF-MLP, 3 seeds, physics-only
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1 2; do
  python3 - "$s" > logs/pilot_vqc_seed$s.log 2>&1 <<'PY' &
import sys, json, os, torch; sys.path.insert(0, '.'); torch.set_num_threads(2)
import qpinn_local as q
s = int(sys.argv[1]); m = q.VQC(6, 4, seed=s)
r = q.train(m, 'exp', 9.0, regime='physics', iters=2000, lr=5e-3, n_c=512, seed=s, log_every=250)
r.update(dict(model='vqc', seed=s, case='exp', gamma=9.0, regime='physics', lr=5e-3, n_q=6, L=4))
json.dump(r, open('results/pilot_vqc_seed%d.json' % s, 'w'), indent=1); print('DONE', r['relL2_T'], r['relL2_sigma'])
PY
done
python3 - > logs/pilot_classical.log 2>&1 <<'PY' &
import sys, json, torch; sys.path.insert(0, '.'); torch.set_num_threads(2)
import qpinn_local as q
target = q.VQC(6, 4).n_params(); out = []
for kind in ('mlp', 'ff'):
    for s in range(3):
        m = q.make_model(kind, target=target, seed=s)
        print('==', kind, 'seed', s, 'params', m.n_params(), flush=True)
        r = q.train(m, 'exp', 9.0, regime='physics', iters=2000, lr=5e-3, n_c=512, seed=s, log_every=250)
        r.update(dict(model=kind, seed=s, case='exp', gamma=9.0, regime='physics', lr=5e-3, target_params=target))
        out.append(r); json.dump(out, open('results/pilot_classical.json', 'w'), indent=1)
        print('DONE', kind, s, r['relL2_T'], r['relL2_sigma'], flush=True)
PY
wait
echo ALL_PILOT_DONE
