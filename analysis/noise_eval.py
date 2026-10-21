"""noise_eval.py -- E4: hardware-realism evaluation of a trained VQC (inference only).
(a) finite shots: each <Z_k> estimated from N_shots Bernoulli samples of the exact probability (1+<Z_k>)/2;
(b) depolarising noise: single-qubit depolarising channel with probability p applied to every qubit after every
    layer, simulated exactly with a density matrix (numpy), for n_q <= 8."""
import numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qpinn_local as q, ground_truth as gt

def gate_matrices(model, u, v):
    """Per-point list over layers of per-qubit fused 2x2 gates (numpy complex), from the torch model."""
    with torch.no_grad():
        B = u.shape[0]; out = []
        for l in range(model.L):
            layer = []
            for k in range(model.n_q):
                G = (q._rz(model.rot[l, k, 2].expand(B)) @ q._ry(model.rot[l, k, 1].expand(B)) @ q._rz(model.rot[l, k, 0].expand(B))
                     @ q._rx(model.enc[l, k, 1] * v) @ q._ry(model.enc[l, k, 0] * u))
                layer.append(G.numpy())
            out.append(layer)
        return out

def _apply_rho(rho, k, G, n):
    # rho: (B, 2^n, 2^n); apply G on qubit k: rho -> (G_k) rho (G_k)^dagger
    B = rho.shape[0]; d = 2 ** n
    r = rho.reshape(B, 2 ** k, 2, 2 ** (n - k - 1), d)
    r = np.einsum('bij,bajcm->baicm', G, r).reshape(B, d, d)
    r = r.reshape(B, d, 2 ** k, 2, 2 ** (n - k - 1))
    r = np.einsum('bij,bmajc->bmaic', G.conj(), r).reshape(B, d, d)
    return r

def _depolarise(rho, k, p, n):
    B = rho.shape[0]; d = 2 ** n
    X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1.0, -1.0]).astype(complex)
    out = (1 - p) * rho
    for P in (X, Y, Z):
        out = out + (p / 3) * _apply_rho(rho, k, np.broadcast_to(P, (B, 2, 2)), n)
    return out

def noisy_expectations(model, x, t, p=0.0, shots=None, rng=None):
    u = 2 * x / q.XMAX - 1; v = 2 * t / q.TMAX - 1
    ut, vt = torch.as_tensor(u), torch.as_tensor(v)
    n = model.n_q; B = len(x); d = 2 ** n
    gates = gate_matrices(model, ut, vt)
    psi0 = np.zeros(d, complex); psi0[0] = 1
    rho = np.broadcast_to(np.outer(psi0, psi0.conj()), (B, d, d)).copy()
    cz = model.cz.numpy()
    for l in range(model.L):
        for k in range(n):
            rho = _apply_rho(rho, k, gates[l][k], n)
        rho = rho * cz[None, :, None] * cz.conj()[None, None, :]
        if p > 0:
            for k in range(n):
                rho = _depolarise(rho, k, p, n)
    probs = np.real(np.einsum('bii->bi', rho))
    zs = model.zsign.numpy()[:, :model.n_out]
    ez = probs @ zs
    if shots is not None:
        rng = rng or np.random.default_rng(0)
        pk = (1 + ez) / 2
        ez = 2 * rng.binomial(shots, np.clip(pk, 0, 1)) / shots - 1
    a = model.out_a.detach().numpy(); b = model.out_b.detach().numpy()
    y = ez * a + b
    # parametrisation of qpinn_local.Fields(param='sigma'):  T = t y0,  sigma = t y1
    T = t * y[:, 0]; S = t * y[:, 1]
    return T, S

def evaluate_noisy(model, gamma, case, p=0.0, shots=None, n=61, seed=0):
    X, Tt = gt.test_grid(n); x = X.ravel(); t = Tt.ravel()
    Te, Se = gt.exact(X, Tt, case, gamma)
    T, S = noisy_expectations(model, x, t, p, shots, np.random.default_rng(seed))
    return dict(p=p, shots=shots, relL2_T=gt.rel_l2(T.reshape(X.shape), Te), relL2_sigma=gt.rel_l2(S.reshape(X.shape), Se))

if __name__ == '__main__':
    m = q.VQC(4, 2)
    X, Tt = gt.test_grid(21); x = X.ravel(); t = Tt.ravel()
    T0, S0 = noisy_expectations(m, x, t)
    with torch.no_grad():
        T1, S1 = q.Fields(m, 9.0, 'exp').sigma(torch.as_tensor(x), torch.as_tensor(t))
    print("density-matrix vs statevector max diff:", np.abs(T0 - T1.numpy()).max(), np.abs(S0 - S1.numpy()).max())
    print(evaluate_noisy(m, 9.0, 'exp', p=0.01, n=21), evaluate_noisy(m, 9.0, 'exp', shots=1000, n=21))
