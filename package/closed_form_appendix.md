# Appendix material: exact solution of the nondimensional laser-pulse thermoelastic problem (verified 20 Oct 2026)

## Problem
T_t = T_xx + e^{-x} f(t), x > 0;  T_x(0,t) = 0;  T(x,0) = 0;  T -> 0 (x -> inf)
sigma_xx - (1/gamma) sigma_tt = T_tt;  sigma(0,t) = 0;  sigma(x,0) = 0;  sigma_t(x,0+) = -gamma e^{-x} f(0+)
(all fields identically zero for t < 0; the initial stress rate follows from u_t = 0 and the constitutive law,
and is produced automatically by the Laplace transform taken from t = 0-).

## Transform-domain solution (fh = Laplace transform of f, a = sqrt(gamma), tau = x/a)
T^(x,s)     = fh(s) [ e^{-x}/(s-1) - e^{-sqrt(s) x} / (sqrt(s)(s-1)) ]
sigma^(x,s) = fh(s) [ A(s)(e^{-x} - e^{-s tau}) - B(s)(e^{-sqrt(s) x} - e^{-s tau}) ],
A(s) = gamma s^2 / ((s-1)(gamma - s^2)),  B(s) = gamma sqrt(s) / ((s-1)(gamma - s)).
The apparent poles at s = +a and s = gamma are removable (the brackets vanish there), so no growing exponential survives; the pole at s = -a is genuine and yields the decaying e^{-a t} term (coefficient r- below).

## Time domain, f = A_e e^{-beta t}  (beta -> 0 gives the unit-step response; superposition gives the step and smooth pulses)
xi = x/(2 sqrt t), eta = sqrt(beta t), c1 = gamma/((gamma-1)(1+beta)), c2 = gamma/((gamma-1)(gamma+beta)),
c3 = -gamma/((1+beta)(gamma+beta)), r+ = -gamma a/(2(a-1)(a+beta)), r- = gamma a/(2(a+1)(a-beta)), rb = gamma beta^2/((1+beta)(beta^2-gamma)).

T/A_e = 1/(1+beta) { e^{t-x}/2 erfc((2t-x)/(2 sqrt t)) + e^{t+x}/2 erfc((2t+x)/(2 sqrt t)) - e^{-x-beta t}
                     - e^{-beta t}/sqrt(beta) Im[ e^{i sqrt(beta) x} erfc(xi + i eta) ] }

sigma/A_e = c1 [ -e^{-xi^2}/sqrt(pi t) + e^{t+x}/2 erfc(xi+sqrt t) + e^{t-x}/2 erfc(sqrt t - xi) ]
          + c2 [  e^{-xi^2}/sqrt(pi t) - a/2 e^{gamma t + a x} erfc(xi + a sqrt t) + a/2 e^{gamma t - a x} erfc(xi - a sqrt t) ]
          + c3 [ -e^{-xi^2}/sqrt(pi t) - sqrt(beta) e^{-beta t} Im( e^{i sqrt(beta) x} erfc(xi + i eta) ) ]
          + rb e^{-beta t - x} + r+ e^{a t - x} + r- e^{-a t - x}
          + H(t - tau) { c1 [ 1/sqrt(pi(t-tau)) - e^{t-tau} erfc(sqrt(t-tau)) ]
                        + c2 [ -1/sqrt(pi(t-tau)) - a e^{gamma(t-tau)} erf(a sqrt(t-tau)) ]
                        + c3 [ 1/sqrt(pi(t-tau)) - sqrt(beta) e^{-beta(t-tau)} erfi(sqrt(beta(t-tau))) ]
                        - rb e^{-beta(t-tau)} - r+ e^{a(t-tau)} - r- e^{-a(t-tau)} }
Unit-step temperature: U_T = -e^{-x} + e^{t-x}/2 erfc((2t-x)/(2 sqrt t)) + e^{t+x}/2 erfc((2t+x)/(2 sqrt t))
                            + 2 sqrt(t/pi) e^{-x^2/4t} - x erfc(x/(2 sqrt t)).
Numerically every e^{...} erfc(...) product is evaluated as e^{-xi^2} erfcx(.), and the e^{gamma t} pieces that cancel
behind the front are regrouped analytically (ground_truth.py).

## Verification
- de Hoog numerical inverse Laplace transform (mpmath, 30 digits) at 25 (x,t) points per pulse (5 depths x 5 times) for the
  exponential and the rectangular pulse, gamma = 9, and at 5 points for the smooth pulse (superposition of two exponential
  responses versus direct inversion of its transform): closed form and inversion agree to the printed 7 significant digits
  at every point (check_corrected.py; log verification_log_closed_form.txt).
- Method-of-lines finite differences (RK4, dx = 0.01, w = sigma + gamma T formulation with w = w_t = 0 at t = 0), same 25
  points per pulse: max |closed form - FD| = 4.20e-6 (T) and 6.18e-3 (sigma) for the exponential pulse and 5.91e-6 (T) and
  6.19e-3 (sigma) for the rectangular pulse; the sigma differences are the FD discretisation error at the front, not a
  closed-form error. These are pointwise checks at the listed points, not a uniform accuracy bound.
- The ground_truth.py self-test compares five points against 30-digit inversion values: worst deviation 4e-10.

## Errors in the printed solution of Al-Qahtani, Lasers in Engineering 61 (2026) (proof dated 18 Sep 2025)
- Eq. (11): second term must be erfc((2t - x)/(2 sqrt t)); as printed it violates T(x,0) = 0.
- Eq. (12): as printed gives T(x,0) = e^{-x}; corrected unit-step response above.
- Eqs. (19)-(26): grow like e^{gamma t}; the corrected stress solution is above.
- Eq. (15) with the stated stress scale sigma_0 = I1 c2 c1^2/(k delta) should carry 1/gamma on T_tt (rescaling only).
- Fig. 2 caption: As*Delta = 0.25 but Ae/beta = 0.5 (energies not equal as stated).
