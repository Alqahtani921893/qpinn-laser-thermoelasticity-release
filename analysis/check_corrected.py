import sys, numpy as np, time
sys.path.insert(0, '.')
import closed_form_corrected as cc, laplace_ref as lr, fd_ref as fd
beta, gamma, Ae, As, Delta = 2.0, 9.0, 1.0, 1.0, 0.5
print("=== corrected closed form vs inverse Laplace (dehoog) and FD, exponential pulse (beta=2, gamma=9) ===")
t_out = [0.1, 0.3, 0.75, 1.5, 2.5]
t0 = time.time(); xg, out = fd.solve(lambda t: Ae*np.exp(-beta*t), Ae, gamma, N=2400, t_end=2.5, t_out=t_out); print("FD N=2400: %.1fs" % (time.time()-t0))
maxT = maxS = 0.0
for t in t_out:
    T, S = out[t]
    for x in [0.1, 0.5, 1.1, 2.1, 4.1]:
        i = int(round(x / (xg[1]-xg[0])))
        Tc = float(cc.T_exp(x, t, beta, Ae)); Sc = float(cc.sigma_exp(x, t, beta, gamma, Ae))
        Tl = lr.T_exp_ref(x, t, beta, Ae, method='dehoog'); Sl = lr.sigma_exp_ref(x, t, beta, gamma, Ae, method='dehoog')
        maxT = max(maxT, abs(Tc-T[i])); maxS = max(maxS, abs(Sc-S[i]))
        print("t=%-5g x=%-4g | T: closed=%-11.7g lap=%-11.7g fd=%-11.7g | sig: closed=%-11.7g lap=%-11.7g fd=%-11.7g" % (t, x, Tc, Tl, T[i], Sc, Sl, S[i]))
print("max |closed - FD|: T %.2e  sigma %.2e" % (maxT, maxS))
print("\n=== step pulse (As=1, Delta=0.5, gamma=9) ===")
xg, out = fd.solve(lambda t: As*(1.0 if t < Delta else 0.0), As, gamma, N=2400, t_end=2.5, t_out=t_out)
maxT = maxS = 0.0
for t in t_out:
    T, S = out[t]
    for x in [0.1, 0.5, 1.1, 2.1, 4.1]:
        i = int(round(x / (xg[1]-xg[0])))
        Tc = float(cc.T_step(x, t, As, Delta)); Sc = float(cc.sigma_step(x, t, As, Delta, gamma))
        Tl = lr.T_step_ref(x, t, As, Delta, method='dehoog'); Sl = lr.sigma_step_ref(x, t, As, Delta, gamma, method='dehoog')
        maxT = max(maxT, abs(Tc-T[i])); maxS = max(maxS, abs(Sc-S[i]))
        print("t=%-5g x=%-4g | T: closed=%-11.7g lap=%-11.7g fd=%-11.7g | sig: closed=%-11.7g lap=%-11.7g fd=%-11.7g" % (t, x, Tc, Tl, T[i], Sc, Sl, S[i]))
print("max |closed - FD|: T %.2e  sigma %.2e" % (maxT, maxS))
print("\n=== large gamma robustness (gamma=100, beta=2): finite values? ===")
for t in [0.05, 0.5, 2.0, 5.0]:
    v = cc.sigma_exp(np.array([0.0, 0.1, 1.0, 5.0, 20.0]), t, beta, 100.0)
    print("t=%g:" % t, np.array2string(np.asarray(v), precision=5))

print("\n=== smooth pulse (A (e^{-2t} - e^{-8t}), A = 4/3) by superposition vs direct de Hoog inversion, gamma = 9 ===")
import mpmath as mp
fh = lambda s: (4.0/3.0) * (1/(s+2.0) - 1/(s+8.0))
mx = 0.0
for (x, t) in [(0.1, 0.1), (1.1, 0.75), (2.1, 1.5), (0.5, 2.5), (4.1, 2.5)]:
    Tl = float(lr.invert(lambda s: lr.That(mp.mpf(x), s, fh), mp.mpf(t), method='dehoog'))
    Sl = float(lr.invert(lambda s: lr.Shat(mp.mpf(x), s, fh, mp.mpf(9.0)), mp.mpf(t), method='dehoog'))
    Tc = float(cc.T_exp(x, t, 2.0, 4/3) - cc.T_exp(x, t, 8.0, 4/3)); Sc = float(cc.sigma_exp(x, t, 2.0, 9.0, 4/3) - cc.sigma_exp(x, t, 8.0, 9.0, 4/3))
    mx = max(mx, abs(Tc - Tl), abs(Sc - Sl))
    print("x=%-4g t=%-5g | T: closed=%-12.8g lap=%-12.8g | sigma: closed=%-12.8g lap=%-12.8g" % (x, t, Tc, Tl, Sc, Sl))
print("max |closed - inversion| over the 5 smooth-pulse points: %.2e" % mx)
