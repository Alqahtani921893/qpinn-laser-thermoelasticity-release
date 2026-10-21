import sys, torch, numpy as np, time; sys.path.insert(0, '.'); torch.set_num_threads(2)
import qpinn_local as q, ground_truth as gt
X, Tt = gt.test_grid(); Te, Se = gt.exact(X, Tt, 'exp', 9.0)
kind, icw, icx, iters, lbfgs = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
m = q.make_model(kind, 6, 4, target=124); t0 = time.time()
r = q.train(m, 'exp', 9.0, regime='physics', iters=iters, lr=3e-3, n_c=1024, seed=0, log_every=1000, sampling='mixed', verbose=True, lbfgs_iters=lbfgs, ic_weight=icw, ic_xmin=icx)
print("RESULT %s icw=%g icx=%g adam=%d lbfgs=%d: relL2 T %.4f sigma %.4f (%.0fs)" % (kind, icw, icx, iters, lbfgs, r['relL2_T'], r['relL2_sigma'], time.time() - t0), flush=True)
