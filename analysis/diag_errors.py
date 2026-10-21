"""Where does the stress error live?  Train the reference MLP (a) data-only with the hard-IC parametrisation,
(b) physics-only, then report relative L2 error of sigma by region of the test grid."""
import sys, torch, numpy as np, time; sys.path.insert(0, '.'); torch.set_num_threads(2)
import qpinn_local as q, ground_truth as gt
X, Tt = gt.test_grid(); Te, Se = gt.exact(X, Tt, 'exp', 9.0)
front = 3.0 * Tt  # x = sqrt(gamma) t
regions = {'early t<0.5': Tt < 0.5, 'mid 0.5<t<1.5': (Tt >= 0.5) & (Tt < 1.5), 'late t>1.5': Tt >= 1.5,
           'ahead x>3t+0.3': X > front + 0.3, 'near front |x-3t|<0.3': np.abs(X - front) < 0.3, 'behind x<3t-0.3': X < front - 0.3,
           'x<1': X < 1, 'x>4': X > 4}
def report(F, tag):
    with torch.no_grad():
        T, S = F.sigma(torch.as_tensor(X.ravel()), torch.as_tensor(Tt.ravel()))
    T = T.numpy().reshape(X.shape); S = S.numpy().reshape(X.shape)
    print("%s: global relL2 T %.4f sigma %.4f" % (tag, gt.rel_l2(T, Te), gt.rel_l2(S, Se)))
    for k, m in regions.items():
        print("   %-24s sigma relL2 %.3f  (share of ||sigma||^2: %.3f)   T relL2 %.3f" % (k, gt.rel_l2(S[m], Se[m]), (Se[m]**2).sum() / (Se**2).sum(), gt.rel_l2(T[m], Te[m])))
for regime, iters, lr, n_d in [('data', 3000, 1e-2, 2048), ('physics', 4000, 3e-3, 0)]:
    m = q.make_model('mlpref'); t0 = time.time()
    r = q.train(m, 'exp', 9.0, regime=regime, iters=iters, lr=lr, n_c=1024, n_d=max(n_d, 1), seed=0, log_every=1000, sampling='mixed', verbose=False)
    report(q.Fields(m, 9.0, 'exp'), "%s (%d its, %.0fs)" % (regime, iters, time.time() - t0))
    torch.save(m.state_dict(), 'results/diag_%s_mlpref.pt' % regime)
