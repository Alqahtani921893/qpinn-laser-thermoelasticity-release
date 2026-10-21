"""Closed-form solutions transcribed from Al-Qahtani, Lasers in Engineering 61 (2026), eqs. (11)-(26).
Nondimensional problem:  T_t = T_xx + e^{-x} f(t),  sigma_xx - (1/gamma) sigma_tt = T_tt,  x >= 0.
All e^{a} erfc(z) products are evaluated as e^{-xi^2} erfcx(z) (identity (xi -/+ c)^2 = xi^2 -/+ 2 xi c + c^2)."""
import numpy as np
from scipy.special import erfc, erfcx

def _xi(x, t):
    return x / (2.0 * np.sqrt(t))

def T_exp(x, t, beta, Ae=1.0):
    """Eq. (11) times Ae (paper writes it for f = e^{-beta t})."""
    x = np.asarray(x, float); t = np.asarray(t, float)
    xi = _xi(x, t); E = np.exp(-xi**2); sb = np.sqrt(beta); eta = np.sqrt(beta * t)
    term = (-np.exp(-x - beta * t) / (1 + beta)
            + E * erfcx(xi - np.sqrt(t)) / (2 * (1 + beta))
            + E * erfcx(xi + np.sqrt(t)) / (2 * (1 + beta))
            - 1j * E * erfcx(xi - 1j * eta) / (2 * sb * (1 + beta))
            + 1j * E * erfcx(xi + 1j * eta) / (2 * sb * (1 + beta)))
    return Ae * np.real(term), Ae * np.imag(term)

def _T_cont(x, t):
    """Unit continuous heating f = H(t): first bracket of eq. (12) (As = 1)."""
    xi = _xi(x, t); E = np.exp(-xi**2)
    return (np.exp(-x) * (1 - np.exp(-t)) + 2 * np.sqrt(t / np.pi) * E - x * erfc(xi)
            + np.exp(t - x) - 0.5 * E * (erfcx(xi - np.sqrt(t)) + erfcx(xi + np.sqrt(t))))

def T_step(x, t, As, Delta):
    x = np.asarray(x, float); t = np.asarray(t, float)
    out = As * _T_cont(x, t)
    m = t > Delta
    if np.any(m):
        out = np.where(m, out - As * _T_cont(x, np.where(m, t - Delta, 1.0)), out)
    return out

def sigma_exp(x, t, beta, gamma, Ae=1.0):
    """Eqs. (18)-(23) times Ae."""
    x = np.asarray(x, float); t = np.asarray(t, float)
    xi = _xi(x, t); E = np.exp(-xi**2); sb = np.sqrt(beta); eta = np.sqrt(beta * t)
    sg = np.sqrt(gamma); spt = np.sqrt(np.pi * t)
    s1 = -np.exp(-x) * (np.exp(-beta * t) / ((1 + beta) * (gamma - 1))
                        - beta**2 * np.exp(-beta * t) / ((1 + beta) * (beta**2 - gamma))
                        + np.exp(t) / ((1 + beta) * (gamma - 1)))
    s2 = E * (2 / spt + erfcx(xi - np.sqrt(t)) - erfcx(xi + np.sqrt(t))) / (2 * (1 + beta) * (1 - gamma))
    s3c = E * (2 / spt + 1j * sb * erfcx(xi - 1j * eta) - 1j * sb * erfcx(xi + 1j * eta)) / (2 * (1 + beta) * (beta + gamma))
    s3 = np.real(s3c)
    s4 = E * (2 / spt + (erfcx(xi - np.sqrt(gamma * t)) - erfcx(xi + np.sqrt(gamma * t))) / sg) / (2 * (gamma - 1) * (beta + gamma))
    tau = t - x / sg
    H = (tau > 0).astype(float)
    tau_c = np.where(tau > 0, tau, 0.0)
    s5 = -sg * (np.exp(-sg * tau_c) / (2 * (sg + 1) * (1 - beta * sg))
                + np.exp(sg * tau_c) / (2 * (sg - 1) * (1 + beta * sg))
                - gamma * np.exp(gamma * tau_c) / ((gamma - 1) * (gamma + beta))
                - np.exp(-beta * tau_c) / ((beta**2 - gamma) * (gamma + beta))) * H
    return Ae * (s1 + s2 + s3 + s4 + s5), dict(s1=Ae*s1, s2=Ae*s2, s3=Ae*s3, s4=Ae*s4, s5=Ae*s5, s3_imag=Ae*np.imag(s3c))

def _S_cont(x, t, As, Delta, gamma):
    """Eq. (26): stress for continuous heating (note the e^{-x-Delta} factor as printed)."""
    xi = _xi(x, t); E = np.exp(-xi**2); sg = np.sqrt(gamma); spt = np.sqrt(np.pi * t)
    F = E * (1 / spt + erfcx(xi - np.sqrt(t)) - erfcx(xi + np.sqrt(t)))
    G = E * (1 / spt + (erfcx(xi - np.sqrt(gamma * t)) - erfcx(xi + np.sqrt(gamma * t))) / sg)
    tau = t - x / sg
    H = (tau > 0).astype(float); tau_c = np.where(tau > 0, tau, 0.0)
    W = 1 / gamma - np.exp(sg * tau_c) / (2 * (sg - 1)) - np.exp(-sg * tau_c) / (2 * (sg + 1)) + gamma * np.exp(gamma * tau_c) / (gamma - 1)
    S = (As / (2 * (1 - gamma)) * F + As / (2 * (gamma - 1)) * G + As * E / (gamma * spt)
         - As * np.exp(-x) * ((np.exp(sg * t) - np.exp(-sg * t)) / (2 * sg * (sg - 1)) - np.exp(t) / (gamma - 1))
         - As * np.exp(-x - Delta) / (gamma - 1) * (np.cosh(t) - np.cosh(sg * t) + np.sinh(t) - np.sinh(sg * t) / sg)
         - As * sg * W * H)
    return S

def sigma_step(x, t, As, Delta, gamma):
    x = np.asarray(x, float); t = np.asarray(t, float)
    out = _S_cont(x, t, As, Delta, gamma)
    m = t > Delta
    if np.any(m):
        out = np.where(m, out - _S_cont(x, np.where(m, t - Delta, 1.0), As, Delta, gamma), out)
    return out
