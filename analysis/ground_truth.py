"""ground_truth.py -- exact solutions of the nondimensional laser-pulse thermoelastic problem
(one-dimensional, semi-infinite solid, Beer-Lambert volumetric source, one-way coupled stress wave).

    T_t = T_xx + e^{-x} f(t),                T_x(0,t) = 0,   T(x,0) = 0,            x >= 0
    sigma_xx - (1/gamma) sigma_tt = T_tt,     sigma(0,t) = 0, sigma(x,0) = 0,
    sigma_t(x,0+) = -gamma e^{-x} f(0+)       (all fields identically zero for t < 0; this is the
                                              initial rate implied by u_t = 0 and the constitutive law)

Pulses:   exp     f = Ae exp(-beta t)
          step    f = As [H(t) - H(t - Delta)]
          smooth  f = A [exp(-beta1 t) - exp(-beta2 t)]      (f(0+) = 0: no impulsive wave launch)

The closed forms below were obtained by Laplace transform (residues + standard erfc tables).  They are
NOT the formulas printed in Al-Qahtani, Lasers in Engineering (2026), eqs. (11), (12), (19)-(26), which
contain sign/coefficient errors (they violate T(x,0) = 0 and grow like exp(gamma t)).  Verification scope (log: verification_log_closed_form.txt): (i) against a 30-digit de Hoog numerical inverse
Laplace transform the corrected forms agree to the printed 7 significant digits at 25 (x, t) points per pulse
for the exponential and the rectangular pulse (gamma = 9), and the smooth-pulse superposition agrees with the
direct inversion of its transform to 5.1e-16 at 5 points; (ii) against a method-of-lines finite-difference
solution (RK4, dx = 0.01) the maxima of |closed form - FD| over the same 25 points are T 4.20e-6 / sigma 6.18e-3
(exponential) and T 5.91e-6 / sigma 6.19e-3 (rectangular), the sigma value being the FD discretisation error at
the wave front, not a closed-form error; (iii) the self-test below regresses five points against stored 30-digit
inversion values (worst deviation 4.0e-10).  These are pointwise checks, not a uniform accuracy bound.
Only numpy and scipy are required.  Run `python ground_truth.py` for the self-test.
"""
import numpy as np
from scipy.special import erfc, erfcx, dawsn

# ----------------------------------------------------------------------------- helpers
def _xi(x, t):
    return x / (2.0 * np.sqrt(t))

def _Eerfcx(xi, u):
    """Stable e^{-xi^2} erfcx(u) for real u of either sign, using erfcx(-u) = 2 e^{u^2} - erfcx(u)."""
    u = np.asarray(u, float)
    direct = np.exp(-xi**2) * erfcx(np.where(u >= 0, u, 0.0))
    alt = 2.0 * np.exp(u**2 - xi**2) - np.exp(-xi**2) * erfcx(np.where(u < 0, -u, 0.0))
    return np.where(u >= 0, direct, alt)

# ----------------------------------------------------------------------------- temperature
def T_exp(x, t, beta, Ae=1.0):
    """f = Ae e^{-beta t}.  Conventional form:
    T = Ae/(1+beta) { e^{t-x}/2 erfc((2t-x)/(2 sqrt t)) + e^{t+x}/2 erfc((2t+x)/(2 sqrt t)) - e^{-x-beta t}
                      - e^{-beta t}/sqrt(beta) Im[ e^{i sqrt(beta) x} erfc(x/(2 sqrt t) + i sqrt(beta t)) ] }"""
    x = np.asarray(x, float); t = np.asarray(t, float)
    xi = _xi(x, t); E = np.exp(-xi**2); eta = np.sqrt(beta * t)
    val = (0.5 * _Eerfcx(xi, np.sqrt(t) - xi) + 0.5 * E * erfcx(xi + np.sqrt(t))
           - np.exp(-x - beta * t)
           - E * np.imag(erfcx(xi + 1j * eta)) / np.sqrt(beta)) / (1.0 + beta)
    return Ae * val

def _U_T(x, t):
    """Unit continuous heating f = H(t):
    U_T = -e^{-x} + e^{t-x}/2 erfc((2t-x)/(2 sqrt t)) + e^{t+x}/2 erfc((2t+x)/(2 sqrt t))
          + 2 sqrt(t/pi) e^{-x^2/(4t)} - x erfc(x/(2 sqrt t))"""
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

def T_smooth(x, t, beta1, beta2, A):
    return T_exp(x, t, beta1, A) - T_exp(x, t, beta2, A)

# ----------------------------------------------------------------------------- stress
def _sigma_unit(x, t, beta, gamma):
    """Stress for f = e^{-beta t} (beta = 0 gives the unit-step response).  With a = sqrt(gamma),
    tau = x/a, xi = x/(2 sqrt t), eta = sqrt(beta t) and
      c1 = g/((g-1)(1+b)), c2 = g/((g-1)(g+b)), c3 = -g/((1+b)(g+b)),
      r+ = -g a/(2(a-1)(a+b)), r- = g a/(2(a+1)(a-b)), rb = g b^2/((1+b)(b^2-g)),
    sigma = c1 [ -e^{-xi^2}/sqrt(pi t) + e^{t+x}/2 erfc(xi+sqrt t) + e^{t-x}/2 erfc(sqrt t - xi) ]
          + c2 [  e^{-xi^2}/sqrt(pi t) - a/2 e^{g t + a x} erfc(xi + a sqrt t) + a/2 e^{g t - a x} erfc(xi - a sqrt t) ]
          + c3 [ -e^{-xi^2}/sqrt(pi t) - sqrt(b) e^{-b t} Im( e^{i sqrt(b) x} erfc(xi + i eta) ) ]
          + rb e^{-b t - x} + r+ e^{a t - x} + r- e^{-a t - x}
          + H(t - tau) { c1 [ 1/sqrt(pi(t-tau)) - e^{t-tau} erfc(sqrt(t-tau)) ]
                        + c2 [ -1/sqrt(pi(t-tau)) - a e^{g(t-tau)} erf(a sqrt(t-tau)) ]
                        + c3 [ 1/sqrt(pi(t-tau)) - sqrt(b) e^{-b(t-tau)} erfi(sqrt(b(t-tau))) ]
                        - rb e^{-b(t-tau)} - r+ e^{a(t-tau)} - r- e^{-a(t-tau)} }.
    The implementation regroups the exponentially growing pieces (which cancel behind the wave front)
    into bounded erfcx terms.  Singular parameter combinations to avoid: gamma = 1, a = beta, beta^2 = gamma."""
    x = np.asarray(x, float); t = np.asarray(t, float)
    a = np.sqrt(gamma); xi = _xi(x, t); E = np.exp(-xi**2); st = np.sqrt(t)
    c1 = gamma / ((gamma - 1) * (1 + beta))
    c2 = gamma / ((gamma - 1) * (gamma + beta))
    c3 = -gamma / ((1 + beta) * (gamma + beta))
    r_a = -gamma * a / (2 * (a - 1) * (a + beta))
    r_ma = gamma * a / (2 * (a + 1) * (a - beta))
    r_b = gamma * beta**2 / ((1 + beta) * (beta**2 - gamma))
    tau = x / a; H = (t > tau); tm = np.where(H, t - tau, 1.0)
    spt = np.sqrt(np.pi * t); sptm = np.sqrt(np.pi * tm)
    g1 = c1 * (-E / spt + 0.5 * E * erfcx(xi + st) + 0.5 * _Eerfcx(xi, st - xi)
               + np.where(H, 1.0 / sptm - erfcx(np.sqrt(tm)), 0.0))
    ahead = c2 * (E / spt - 0.5 * a * E * erfcx(xi + a * st) + 0.5 * a * _Eerfcx(xi, xi - a * st))
    behind = c2 * (E / spt - 0.5 * a * E * erfcx(xi + a * st) - 0.5 * a * _Eerfcx(xi, a * st - xi)
                   + a * erfcx(a * np.sqrt(tm)) - 1.0 / sptm)
    g2 = np.where(H, behind, ahead)
    eta = np.sqrt(beta * t); etam = np.sqrt(beta * tm)
    g3 = (-c3 * E / spt - c3 * np.sqrt(beta) * E * np.imag(erfcx(xi + 1j * eta))
          + r_b * np.exp(-beta * t - x)
          + np.where(H, c3 / sptm - c3 * np.sqrt(beta) * (2 / np.sqrt(np.pi)) * dawsn(etam)
                     - r_b * np.exp(-beta * tm), 0.0))
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

def sigma_smooth(x, t, beta1, beta2, A, gamma):
    return sigma_exp(x, t, beta1, gamma, A) - sigma_exp(x, t, beta2, gamma, A)

# ----------------------------------------------------------------------------- benchmark cases
CASES = {
    # equal pulse energy E0 = 0.5 in every case
    "exp":    dict(kind="exp",    beta=2.0, Ae=1.0),
    "step":   dict(kind="step",   As=1.0, Delta=0.5),
    "smooth": dict(kind="smooth", beta1=2.0, beta2=8.0, A=4.0/3.0),
}
DOMAIN = dict(x_max=6.0, t_max=3.0)
GAMMA_MAIN = 9.0
GAMMA_SECONDARY = (25.0, 49.0)

def pulse(t, case):
    c = CASES[case] if isinstance(case, str) else case
    t = np.asarray(t, float)
    if c["kind"] == "exp":
        return c["Ae"] * np.exp(-c["beta"] * t) * (t >= 0)
    if c["kind"] == "step":
        return c["As"] * ((t >= 0) & (t < c["Delta"])).astype(float)
    return c["A"] * (np.exp(-c["beta1"] * t) - np.exp(-c["beta2"] * t)) * (t >= 0)

def exact(x, t, case, gamma=GAMMA_MAIN):
    """Return (T, sigma) for the named case at points x, t (t > 0)."""
    c = CASES[case] if isinstance(case, str) else case
    if c["kind"] == "exp":
        return T_exp(x, t, c["beta"], c["Ae"]), sigma_exp(x, t, c["beta"], gamma, c["Ae"])
    if c["kind"] == "step":
        return T_step(x, t, c["As"], c["Delta"]), sigma_step(x, t, c["As"], c["Delta"], gamma)
    return (T_smooth(x, t, c["beta1"], c["beta2"], c["A"]),
            sigma_smooth(x, t, c["beta1"], c["beta2"], c["A"], gamma))

def test_grid(n=241, x_max=DOMAIN["x_max"], t_max=DOMAIN["t_max"]):
    """Fixed evaluation grid: n x-nodes on [0, x_max], n-1 t-nodes on (0, t_max] (t = 0 excluded)."""
    x = np.linspace(0.0, x_max, n)
    t = np.linspace(0.0, t_max, n)[1:]
    X, Tt = np.meshgrid(x, t, indexing="ij")
    return X, Tt

def rel_l2(pred, ref):
    pred = np.asarray(pred, float); ref = np.asarray(ref, float)
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

# ----------------------------------------------------------------------------- self-test
_REFERENCE = [  # (x, t, T_exp, sigma_exp, T_step, sigma_step) from a 30-digit de Hoog inverse Laplace transform,
                # beta=2, Ae=1 | As=1, Delta=0.5 | gamma=9
    (0.1, 0.1, 0.07149377465, -0.1601574284, 0.07917932781, -0.1951384832),
    (1.1, 0.3, 0.08442477929, -0.8394758361, 0.1114338174, -1.096760078),
    (2.1, 0.75, 0.07008054175, -0.7143076852, 0.09276267359, -1.088576979),
    (0.5, 1.5, 0.195006974, 0.08181417056, 0.1925034837, 0.05845862163),
    (4.1, 2.5, 0.0369710468, 0.231600584, 0.04057057671, 0.2126974832),
]

if __name__ == "__main__":
    worst = 0.0
    for x, t, Te, Se, Ts, Ss in _REFERENCE:
        d = [abs(float(T_exp(x, t, 2.0)) - Te), abs(float(sigma_exp(x, t, 2.0, 9.0)) - Se),
             abs(float(T_step(x, t, 1.0, 0.5)) - Ts), abs(float(sigma_step(x, t, 1.0, 0.5, 9.0)) - Ss)]
        worst = max(worst, max(d))
        print("x=%-4g t=%-5g  max abs dev from reference = %.1e" % (x, t, max(d)))
    X, Tt = test_grid()
    for case in CASES:
        T, S = exact(X, Tt, case)
        assert np.all(np.isfinite(T)) and np.all(np.isfinite(S)), case
        print("case %-6s  grid %s  |T|max=%.4f  |sigma|max=%.4f" % (case, T.shape, np.abs(T).max(), np.abs(S).max()))
    print("SELF-TEST", "PASSED" if worst < 1e-9 else "FAILED", "(worst deviation %.1e)" % worst)
