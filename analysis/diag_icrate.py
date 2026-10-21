import sys, torch, numpy as np, time; sys.path.insert(0, '.'); torch.set_num_threads(2)
import qpinn_local as q, ground_truth as gt
X, Tt = gt.test_grid(); Te, Se = gt.exact(X, Tt, 'exp', 9.0); front = 3.0 * Tt
kind, iters, lbfgs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = q.make_model(kind, 6, 4, target=124); t0 = time.time()
r = q.train(m, 'exp', 9.0, regime='physics', iters=iters, lr=3e-3 if kind != 'vqc' else 5e-3, n_c=1024 if kind != 'vqc' else 512, seed=0, log_every=500, sampling='mixed', verbose=True, lbfgs_iters=lbfgs)
F = q.Fields(m, 9.0, 'exp')
with torch.no_grad(): T, S = F.sigma(torch.as_tensor(X.ravel()), torch.as_tensor(Tt.ravel()))
T = T.numpy().reshape(X.shape); S = S.numpy().reshape(X.shape)
print("icrate %s (%d params): relL2 T %.4f sigma %.4f (%.0fs)" % (kind, m.n_params(), gt.rel_l2(T, Te), gt.rel_l2(S, Se), time.time() - t0))
for k, msk in {'near front |x-3t|<0.3': np.abs(X - front) < 0.3, 'ahead': X > front + 0.3, 'behind': X < front - 0.3, 'early t<0.5': Tt < 0.5, 'x<1': X < 1}.items():
    print("   %-24s sigma relL2 %.3f   T relL2 %.3f" % (k, gt.rel_l2(S[msk], Se[msk]), gt.rel_l2(T[msk], Te[msk])))
torch.save(m.state_dict(), 'results/diag_icrate_%s.pt' % kind)
