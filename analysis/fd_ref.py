"""Independent reference 2: method-of-lines finite differences (RK4) for the coupled system,
using w = sigma + gamma*T so that T_tt never appears:  T_t = T_xx + e^{-x} f,  w_tt = gamma (w - gamma T)_xx.
BCs: T_x(0,t)=0, w(0,t)=gamma*T(0,t) (free surface sigma=0), far field zero.
ICs: T=w=w_t=0.  (w = sigma + gamma*T is the scaled elastic strain; u_t = 0 at t=0 gives w_t(x,0+) = 0, which is NOT a zero stress rate:
 i.e. sigma_t(x,0+) = -gamma e^{-x} f(0+), the same jump the 0- Laplace transform of T_tt produces.)"""
import numpy as np

def solve(f, f0, gamma, L=24.0, N=1200, t_end=3.0, t_out=None, cfl=0.2):
    x = np.linspace(0.0, L, N + 1); dx = x[1] - x[0]
    dt = cfl * dx**2
    nsteps = int(np.ceil(t_end / dt)); dt = t_end / nsteps
    ex = np.exp(-x)
    T = np.zeros(N + 1); w = np.zeros(N + 1); p = np.zeros(N + 1)

    def rhs(t, T, w, p):
        Txx = np.zeros_like(T)
        Txx[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / dx**2
        Txx[0] = 2 * (T[1] - T[0]) / dx**2
        dT = Txx + ex * f(t); dT[-1] = 0.0
        w = w.copy(); w[0] = gamma * T[0]; w[-1] = 0.0
        wxx = np.zeros_like(w)
        wxx[1:-1] = (w[2:] - 2 * w[1:-1] + w[:-2]) / dx**2
        dp = gamma * wxx - gamma**2 * Txx; dp[0] = 0.0; dp[-1] = 0.0
        dw = p.copy(); dw[0] = gamma * dT[0]; dw[-1] = 0.0
        return dT, dw, dp

    t = 0.0; out = {}
    t_out = sorted(t_out or [])
    k_out = {int(round(tt / dt)): tt for tt in t_out}
    for n in range(nsteps):
        k1 = rhs(t, T, w, p)
        k2 = rhs(t + dt/2, T + dt/2*k1[0], w + dt/2*k1[1], p + dt/2*k1[2])
        k3 = rhs(t + dt/2, T + dt/2*k2[0], w + dt/2*k2[1], p + dt/2*k2[2])
        k4 = rhs(t + dt, T + dt*k3[0], w + dt*k3[1], p + dt*k3[2])
        T = T + dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        w = w + dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        p = p + dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        w[0] = gamma * T[0]; w[-1] = 0.0; T[-1] = 0.0
        t += dt
        if (n + 1) in k_out:
            out[k_out[n + 1]] = (T.copy(), (w - gamma * T).copy())
    return x, out
