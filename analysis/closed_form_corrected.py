"""Re-derived closed-form solutions (Laplace transform, residues + standard erfc tables) for
    T_t = T_xx + e^{-x} f(t),  T_x(0,t)=0,  T(x,0)=0
    sigma_xx - (1/gamma) sigma_tt = T_tt,  sigma(0,t)=0,  sigma(x,0) = 0 and sigma_t(x,0+) = -gamma e^{-x} f(0+)
    (the impulsive launch of the stress wave; equivalently w = sigma + gamma T has w = w_t = 0 at t = 0)
with f = Ae e^{-beta t} (exponential pulse) or f = As [H(t) - H(t-Delta)] (step pulse).
Transform-domain solution (a = sqrt(gamma), tau = x/a):
    T^ = fh [ e^{-x}/(s-1) - e^{-sqrt(s) x}/(sqrt(s)(s-1)) ]
    S^ = fh [ A(s)(e^{-x} - e^{-s tau}) - B(s)(e^{-sqrt(s) x} - e^{-s tau}) ],
    A = gamma s^2/((s-1)(gamma-s^2)),  B = gamma sqrt(s)/((s-1)(gamma-s)).
All exponentially growing pieces are regrouped so that only bounded erfcx/erfc terms remain."""
import numpy as np
from scipy.special import erfc, erfcx, dawsn

def _xi(x, t):
    return x / (2.0 * np.sqrt(t))

def _Eerfcx(xi, u):
    """Stable e^{-xi^2} erfcx(u) for real u of either sign (erfcx(-u) = 2 e^{u^2} - erfcx(u));
    valid whenever e^{u^2 - xi^2} is bounded, which holds for every use below."""
    u = np.asarray(u, float)
    direct = np.exp(-xi**2) * erfcx(np.where(u >= 0, u, 0.0))
    alt = 2.0 * np.exp(u**2 - xi**2) - np.exp(-xi**2) * erfcx(np.where(u < 0, -u, 0.0))
    return np.where(u >= 0, direct, alt)

def T_exp(x, t, beta, Ae=1.0):
    """Corrected eq. (11).  Differs from the printed one only in the sign of the argument of the
    second erfc: erfc((2t - x)/(2 sqrt t)) instead of erfc((x - 2t)/(2 sqrt t))."""
    x = np.asarray(x, float); t = np.asarray(t, float)
    xi = _xi(x, t); E = np.exp(-xi**2); eta = np.sqrt(beta * t)
    val = (0.5 * _Eerfcx(xi, np.sqrt(t) - xi) + 0.5 * E * erfcx(xi + np.sqrt(t))
           - np.exp(-x - beta * t)
           - E * np.imag(erfcx(xi + 1j * eta)) / np.sqrt(beta)) / (1.0 + beta)
    return Ae * val

def _U_T(x, t):
    """Unit-step (continuous heating, f = H(t)) temperature."""
    xi = _xi(x, t); E = np.exp(-xi**2)
    return (-np.exp(-x) + 0.5 * _Eerfcx(xi, np.sqrt(t) - xi) + 0.5 * E * erfcx(xi + np.sqrt(t))
            + 2.0 * np.sqrt(t / np.pi) * E - x * erfc(xi))

def T_step(x, t, As, Delta):
    x = np.asarray(x, float); t = np.asarray(t, float)
    out = As * _U_T(x, t)
    m = t > Delta
    if np.any(m):
        out = np.where(m, out - As * _U_T(x, np.where(m, t - Delta, 1.0)), out)
    return out

def _sigma_unit(x, t, beta, gamma):
    """Stress for f = e^{-beta t} (beta = 0 gives the unit-step response)."""
    x = np.asarray(x, float); t = np.asarray(t, float)
    a = np.sqrt(gamma); xi = _xi(x, t); E = np.exp(-xi**2); st = np.sqrt(t)
    c1 = gamma / ((gamma - 1) * (1 + beta))
    c2 = gamma / ((gamma - 1) * (gamma + beta))
    c3 = -gamma / ((1 + beta) * (gamma + beta))
    r_a = -gamma * a / (2 * (a - 1) * (a + beta))
    r_ma = gamma * a / (2 * (a + 1) * (a - beta))
    r_b = gamma * beta**2 / ((1 + beta) * (beta**2 - gamma))
    tau = x / a; H = (t > tau); tm = np.where(H, t - tau, 1.0)   # tm only used where H
    spt = np.sqrt(np.pi * t); sptm = np.sqrt(np.pi * tm)
    # group p = 1
    g1 = c1 * (-E / spt + 0.5 * E * erfcx(xi + st) + 0.5 * _Eerfcx(xi, st - xi)
               + np.where(H, 1.0 / sptm - erfcx(np.sqrt(tm)), 0.0))
    # group p = gamma
    ahead = c2 * (E / spt - 0.5 * a * E * erfcx(xi + a * st) + 0.5 * a * _Eerfcx(xi, xi - a * st))
    behind = c2 * (E / spt - 0.5 * a * E * erfcx(xi + a * st) - 0.5 * a * _Eerfcx(xi, a * st - xi)
                   + a * erfcx(a * np.sqrt(tm)) - 1.0 / sptm)
    g2 = np.where(H, behind, ahead)
    # group p = -beta
    eta = np.sqrt(beta * t); etam = np.sqrt(beta * tm)
    g3 = (-c3 * E / spt - c3 * np.sqrt(beta) * E * np.imag(erfcx(xi + 1j * eta))
          + r_b * np.exp(-beta * t - x)
          + np.where(H, c3 / sptm - c3 * np.sqrt(beta) * (2 / np.sqrt(np.pi)) * dawsn(etam)
                     - r_b * np.exp(-beta * tm), 0.0))
    # +-a exponentials
    g4 = r_a * np.exp(a * t - x) * (~H) + r_ma * (np.exp(-a * t - x) - np.where(H, np.exp(x - a * t), 0.0))
    return g1 + g2 + g3 + g4

def sigma_exp(x, t, beta, gamma, Ae=1.0):
    return Ae * _sigma_unit(x, t, beta, gamma)

def sigma_step(x, t, As, Delta, gamma):
    x = np.asarray(x, float); t = np.asarray(t, float)
    out = As * _sigma_unit(x, t, 0.0, gamma)
    m = t > Delta
    if np.any(m):
        out = np.where(m, out - As * _sigma_unit(x, np.where(m, t - Delta, 1.0), 0.0, gamma), out)
    return out
