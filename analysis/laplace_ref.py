"""Independent reference: numerical inverse Laplace transform (mpmath) of the exact transform-domain solution.
T^(x,s) = fhat(s) [ e^{-x}/(s-1) - e^{-sqrt(s) x}/(sqrt(s)(s-1)) ]          (T_x(0,t)=0, bounded)
sigma^ = s^2 C1 (e^{-x} - e^{-k x})/(1-k^2) + s^2 C2 (e^{-sqrt(s) x} - e^{-k x})/(s-k^2),  k = s/sqrt(gamma)
with C1 = fhat/(s-1), C2 = -fhat/(sqrt(s)(s-1));  sigma(0,t)=0 (free surface), zero initial data."""
import mpmath as mp

def That(x, s, fhat):
    return fhat(s) * (mp.e**(-x) / (s - 1) - mp.e**(-mp.sqrt(s) * x) / (mp.sqrt(s) * (s - 1)))

def Shat(x, s, fhat, gamma):
    k = s / mp.sqrt(gamma)
    C1 = fhat(s) / (s - 1)
    C2 = -fhat(s) / (mp.sqrt(s) * (s - 1))
    return (s**2 * C1 * (mp.e**(-x) - mp.e**(-k * x)) / (1 - k**2)
            + s**2 * C2 * (mp.e**(-mp.sqrt(s) * x) - mp.e**(-k * x)) / (s - k**2))

def invert(Fs, t, method='talbot', dps=30):
    mp.mp.dps = dps
    return mp.invertlaplace(Fs, t, method=method)

def T_exp_ref(x, t, beta, Ae=1.0, **kw):
    fhat = lambda s: Ae / (s + beta)
    return float(invert(lambda s: That(mp.mpf(x), s, fhat), mp.mpf(t), **kw))

def sigma_exp_ref(x, t, beta, gamma, Ae=1.0, **kw):
    fhat = lambda s: Ae / (s + beta)
    return float(invert(lambda s: Shat(mp.mpf(x), s, fhat, mp.mpf(gamma)), mp.mpf(t), **kw))

def _unit_step(x, t, gamma, which, **kw):
    fhat = lambda s: 1 / s
    if which == 'T':
        return float(invert(lambda s: That(mp.mpf(x), s, fhat), mp.mpf(t), **kw))
    return float(invert(lambda s: Shat(mp.mpf(x), s, fhat, mp.mpf(gamma)), mp.mpf(t), **kw))

def T_step_ref(x, t, As, Delta, **kw):
    v = As * _unit_step(x, t, None, 'T', **kw)
    if t > Delta:
        v -= As * _unit_step(x, t - Delta, None, 'T', **kw)
    return v

def sigma_step_ref(x, t, As, Delta, gamma, **kw):
    v = As * _unit_step(x, t, gamma, 'S', **kw)
    if t > Delta:
        v -= As * _unit_step(x, t - Delta, gamma, 'S', **kw)
    return v
