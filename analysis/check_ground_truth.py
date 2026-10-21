import sys, numpy as np
sys.path.insert(0, '.')
import closed_form as cf, laplace_ref as lr, fd_ref as fd
np.set_printoptions(precision=6, suppress=True, linewidth=150)
beta, gamma, Ae = 2.0, 9.0, 1.0
As, Delta = 1.0, 0.5
pts_x = [0.1, 0.5, 1.1, 2.1]; pts_t = [0.1, 0.3, 0.75, 1.5, 2.5]
print("=== Exponential pulse: beta=%g gamma=%g Ae=%g ===" % (beta, gamma, Ae))
print("x      t      T_closed      T_laplace     sigma_closed  sigma_laplace  (imag parts of closed T, s3)")
for t in pts_t:
    for x in pts_x:
        Tc, Tim = cf.T_exp(x, t, beta, Ae); Tl = lr.T_exp_ref(x, t, beta, Ae)
        Sc, parts = cf.sigma_exp(x, t, beta, gamma, Ae); Sl = lr.sigma_exp_ref(x, t, beta, gamma, Ae)
        print("%-6g %-6g %-13.6g %-13.6g %-13.6g %-13.6g  im(T)=%.1e im(s3)=%.1e" % (x, t, Tc, Tl, Sc, Sl, Tim, parts['s3_imag']))
print("\n=== Step pulse: As=%g Delta=%g gamma=%g ===" % (As, Delta, gamma))
print("x      t      T_closed      T_laplace     sigma_closed  sigma_laplace")
for t in pts_t:
    for x in pts_x:
        Tc = float(cf.T_step(x, t, As, Delta)); Tl = lr.T_step_ref(x, t, As, Delta)
        Sc = float(cf.sigma_step(x, t, As, Delta, gamma)); Sl = lr.sigma_step_ref(x, t, As, Delta, gamma)
        print("%-6g %-6g %-13.6g %-13.6g %-13.6g %-13.6g" % (x, t, Tc, Tl, Sc, Sl))
print("\n=== FD cross-check of the Laplace reference (exponential pulse) ===")
xg, out = fd.solve(lambda t: Ae*np.exp(-beta*t), Ae, gamma, t_end=1.5, t_out=[0.3, 0.75, 1.5])
for t in [0.3, 0.75, 1.5]:
    T, S = out[t]
    for x in [0.1, 1.1, 2.1]:
        i = int(round(x / (xg[1]-xg[0])))
        print("t=%-5g x=%-4g  T_fd=%-12.6g T_lap=%-12.6g  sig_fd=%-12.6g sig_lap=%-12.6g" % (t, x, T[i], lr.T_exp_ref(x,t,beta,Ae), S[i], lr.sigma_exp_ref(x,t,beta,gamma,Ae)))
print("\n=== FD cross-check (step pulse) ===")
xg, out = fd.solve(lambda t: As*(1.0 if t < Delta else 0.0), As, gamma, t_end=1.5, t_out=[0.3, 0.75, 1.5])
for t in [0.3, 0.75, 1.5]:
    T, S = out[t]
    for x in [0.1, 1.1, 2.1]:
        i = int(round(x / (xg[1]-xg[0])))
        print("t=%-5g x=%-4g  T_fd=%-12.6g T_lap=%-12.6g  sig_fd=%-12.6g sig_lap=%-12.6g" % (t, x, T[i], lr.T_step_ref(x,t,As,Delta), S[i], lr.sigma_step_ref(x,t,As,Delta,gamma)))
