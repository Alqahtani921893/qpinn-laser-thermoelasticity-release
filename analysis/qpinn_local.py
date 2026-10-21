"""qpinn_local.py -- independent local implementation of the QPINN benchmark (PyTorch, exact statevector).
Models: VQC (data re-uploading, n_q qubits, L layers), MLP, Fourier-feature MLP; class Fields maps the two network
outputs to the physical fields.  Default parametrisation (param='sigma', used for every reported run): hard initial
fields T = t*y0, sigma = t*y1.  Physics loss: heat residual T_t - T_xx - e^{-x} f(t), stress residual
sigma_xx - sigma_tt/gamma - T_tt, surface conditions T_x(0,t) = 0 and sigma(0,t) = 0, exact Dirichlet data at x = x_max
(domain truncation), and soft initial-rate terms on the collocation points with x >= ic_xmin (default 0.3).  The 'w'
(w = sigma + gamma T) and 'taylor' parametrisations are superseded variants kept for the record.
Usage: python qpinn_local.py run|e2 [options]   (see argparse below)."""
import math, json, time, argparse, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'chatgpt_pro_job', 'inputs'))
import ground_truth as gt
torch.set_default_dtype(torch.float64)
XMAX, TMAX = gt.DOMAIN['x_max'], gt.DOMAIN['t_max']

# ----------------------------------------------------------------------------- quantum model
def _ry(th):
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex128)
def _rx(th):
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    z = torch.zeros_like(c)
    re = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
    im = torch.stack([torch.stack([z, -s], -1), torch.stack([-s, z], -1)], -2)
    return torch.complex(re, im)
def _rz(th):
    e = torch.exp(-0.5j * th.to(torch.complex128)); z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)

class VQC(torch.nn.Module):
    """Data re-uploading circuit: per layer, per qubit RY(a*u) RX(b*v) then RZ RY RZ; CZ ring; <Z_k> outputs."""
    def __init__(self, n_q=6, L=4, n_out=2, seed=0, n_in=2):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_q, self.L, self.n_out, self.n_in = n_q, L, n_out, n_in
        self.enc = torch.nn.Parameter(torch.randn(L, n_q, n_in, generator=g))
        self.rot = torch.nn.Parameter(torch.rand(L, n_q, 3, generator=g) * 2 * math.pi)
        self.out_a = torch.nn.Parameter(torch.ones(n_out)); self.out_b = torch.nn.Parameter(torch.zeros(n_out))
        idx = torch.arange(2 ** n_q)
        bits = ((idx[:, None] >> (n_q - 1 - torch.arange(n_q))[None, :]) & 1)       # qubit 0 = most significant
        self.register_buffer('zsign', (1 - 2 * bits).to(torch.float64))               # (2^n, n_q)
        cz = torch.ones(2 ** n_q)
        pairs = [(q, (q + 1) % n_q) for q in range(n_q)] if n_q > 2 else ([(0, 1)] if n_q == 2 else [])
        for q1, q2 in pairs:
            cz = cz * (1 - 2 * (bits[:, q1] * bits[:, q2]).to(torch.float64))
        self.register_buffer('cz', cz.to(torch.complex128))
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
    def _apply(self, psi, q, M):
        B = psi.shape[0]; n = self.n_q
        p = psi.reshape(B, 2 ** q, 2, 2 ** (n - q - 1))
        if M.dim() == 2:
            M = M.expand(B, 2, 2)
        return torch.matmul(M[:, None, :, :], p).reshape(B, 2 ** n)
    def forward(self, u, v, phi=None):
        B = u.shape[0]
        psi = torch.zeros(B, 2 ** self.n_q, dtype=torch.complex128); psi[:, 0] = 1.0
        for l in range(self.L):
            for q in range(self.n_q):
                # fused single-qubit gate: RZ(r3) RY(r2) RZ(r1) [RY(c phi)] RX(b v) RY(a u)   (rightmost acts first)
                G = _rz(self.rot[l, q, 2].expand(B)) @ _ry(self.rot[l, q, 1].expand(B)) @ _rz(self.rot[l, q, 0].expand(B))
                if self.n_in == 3:
                    G = G @ _ry(self.enc[l, q, 2] * phi)
                G = G @ _rx(self.enc[l, q, 1] * v) @ _ry(self.enc[l, q, 0] * u)
                psi = self._apply(psi, q, G)
            psi = psi * self.cz
        prob = (psi.real ** 2 + psi.imag ** 2)
        z = prob @ self.zsign[:, :self.n_out]                                            # (B, n_out)
        return z * self.out_a + self.out_b

# ----------------------------------------------------------------------------- classical models
class MLP(torch.nn.Module):
    def __init__(self, width=8, depth=2, n_out=2, seed=0, n_in=2):
        super().__init__(); torch.manual_seed(seed)
        layers, d = [], n_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]; d = width
        layers += [torch.nn.Linear(d, n_out)]
        self.net = torch.nn.Sequential(*layers)
    def n_params(self): return sum(p.numel() for p in self.parameters())
    def forward(self, u, v, phi=None):
        return self.net(torch.stack([u, v] + ([phi] if phi is not None else []), -1))

class FFMLP(torch.nn.Module):
    """Random Fourier features (fixed) followed by an MLP."""
    def __init__(self, m=8, width=6, depth=1, n_out=2, seed=0, scale=1.0, n_in=2):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.register_buffer('Bf', torch.randn(n_in, m, generator=g) * scale * math.pi)
        self.mlp = MLP(width=width, depth=depth, n_out=n_out, seed=seed, n_in=2 * m)
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def forward(self, u, v, phi=None):
        z = torch.stack([u, v] + ([phi] if phi is not None else []), -1) @ self.Bf
        return self.mlp.net(torch.cat([torch.cos(z), torch.sin(z)], -1))

def matched_mlp(target, depth=2, seed=0, n_in=2):
    best = None
    for w in range(2, 400):
        n = MLP(w, depth, seed=seed, n_in=n_in).n_params()
        if best is None or abs(n - target) < abs(best[1] - target): best = (w, n)
    return MLP(best[0], depth, seed=seed, n_in=n_in)

def matched_ff(target, m=8, depth=1, seed=0, n_in=2):
    best = None
    for w in range(2, 400):
        n = FFMLP(m, w, depth, seed=seed, n_in=n_in).n_params()
        if best is None or abs(n - target) < abs(best[1] - target): best = (w, n)
    return FFMLP(m, best[0], depth, seed=seed, n_in=n_in)

# ----------------------------------------------------------------------------- fields and losses
class Fields:
    """Network outputs -> physical fields with hard initial conditions.
    param='sigma' (default): T = t*y0, sigma = t*y1  (sigma_t(x,0) = y1(x,0) can match -gamma e^{-x} f(0+)).
    param='w':               T = t*y0, w = t^2*y1, sigma = w - gamma*T  (superseded: the gamma*T cancellation
                             made sigma errors feed back into T; kept for the record)."""
    def __init__(self, model, gamma, case, param='sigma', front=False, kappa=3.0):
        self.m, self.gamma, self.case, self.param, self.front, self.kappa = model, gamma, case, param, front, kappa
        self.f0 = float(gt.pulse(np.array([0.0]), case)[0])   # f(0+)
    def sigma(self, x, t):
        u = 2 * x / XMAX - 1; v = 2 * t / TMAX - 1
        if self.front:   # smooth characteristic coordinate: which side of the wave front x = sqrt(gamma) t
            phi = torch.tanh(self.kappa * (t - x / math.sqrt(self.gamma)))
            y = self.m(u, v, phi)
        else:
            y = self.m(u, v)
        if self.param == 'taylor':
            # hard initial conditions including the initial rates:  T_t(x,0) = e^{-x} f(0+)  (heat eq. at t=0)
            # and sigma_t(x,0+) = -gamma e^{-x} f(0+)  (impulsive launch of the stress wave); the network learns the O(t^2) rest.
            ex = torch.exp(-x) * self.f0
            return t * ex + t ** 2 * y[:, 0], -self.gamma * t * ex + t ** 2 * y[:, 1]
        T = t * y[:, 0]
        if self.param == 'sigma':
            # sigma = t y1 (unscaled; the gamma-scaled variant sigma = gamma t y1 started the circuits far from the
            # solution and trained worse at equal budget, see STATUS.md 20 Oct)
            return T, t * y[:, 1]
        return T, t ** 2 * y[:, 1] - self.gamma * T
    def __call__(self, x, t):
        T, S = self.sigma(x, t); return T, S + self.gamma * T

def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

def physics_loss(F, xc, tc, tb, Tb6, Sb6, weights=(1.0, 1.0, 1.0), w_scale=None, ic_weight=1.0, ic_xmin=0.3):
    xc = xc.requires_grad_(True); tc = tc.requires_grad_(True)
    T, S = F.sigma(xc, tc)
    T_t = grad(T, tc); T_tt = grad(T_t, tc); T_x = grad(T, xc); T_xx = grad(T_x, xc)
    S_t = grad(S, tc); S_tt = grad(S_t, tc); S_x = grad(S, xc); S_xx = grad(S_x, xc)
    f = torch.as_tensor(gt.pulse(tc.detach().numpy(), F.case))
    r_h = T_t - T_xx - torch.exp(-xc) * f                       # heat equation
    r_w = S_xx - S_tt / F.gamma - T_tt                           # stress wave equation, eq. (15) form
    if w_scale is not None: r_w = r_w / w_scale
    # surface x = 0
    x0 = torch.zeros_like(tb).requires_grad_(True); t0 = tb.clone()
    T0, s0 = F.sigma(x0, t0); T0_x = grad(T0, x0)
    # far boundary x = XMAX (exact Dirichlet data)
    x6 = torch.full_like(tb, XMAX); T6, s6 = F.sigma(x6, tb)
    # initial rates at t = 0 (soft):  T_t(x,0) = e^{-x} f(0+),  sigma_t(x,0+) = -gamma e^{-x} f(0+)
    # (with T = t*y0, sigma = t*y1 these are conditions on y(x, t=0); the sigma rate is what makes the stress problem well posed)
    li = torch.zeros(())
    if F.param == 'sigma' and ic_weight > 0:
        # the exact rates are discontinuous at the corner (x, t) = (0, 0) (free surface), so the rate condition is
        # imposed only for x >= ic_xmin
        xi = xc.detach(); xi = xi[xi >= ic_xmin]
        yi = F.m(2 * xi / XMAX - 1, torch.full_like(xi, -1.0), *([torch.tanh(F.kappa * (-xi / math.sqrt(F.gamma)))] if F.front else []))
        ex = torch.exp(-xi) * F.f0
        li = ic_weight * (((yi[:, 0] - ex) ** 2).mean() + (((yi[:, 1] + F.gamma * ex) / F.gamma) ** 2).mean())
    lp = (r_h ** 2).mean() + (r_w ** 2).mean()
    lb = (T0_x ** 2).mean() + (s0 ** 2).mean() + ((T6 - Tb6) ** 2).mean() + ((s6 - Sb6) ** 2).mean() + li
    return weights[0] * lp + weights[1] * lb, dict(pde=lp.item(), bc=lb.item(), ic=float(li.detach()))

def data_loss(F, xd, td, Td, Sd):
    T, S = F.sigma(xd, td)
    return ((T - Td) ** 2).mean() + ((S - Sd) ** 2).mean()

# ----------------------------------------------------------------------------- evaluation
def evaluate(F, gamma, case, n=241):
    X, Tt = gt.test_grid(n)
    Te, Se = gt.exact(X, Tt, case, gamma)
    with torch.no_grad():
        T, S = F.sigma(torch.as_tensor(X.ravel()), torch.as_tensor(Tt.ravel()))
    T = T.numpy().reshape(X.shape); S = S.numpy().reshape(X.shape)
    return dict(relL2_T=gt.rel_l2(T, Te), relL2_sigma=gt.rel_l2(S, Se),
                maxabs_T=float(np.abs(T - Te).max()), maxabs_sigma=float(np.abs(S - Se).max()))

# ----------------------------------------------------------------------------- training
def sample_colloc(rng, n, mode='mixed'):
    """Collocation sampling.  'uniform': uniform on the domain.  'mixed': half uniform, half concentrated
    near the heated corner and the early times (x ~ Exp(1), t ~ Exp(0.7), truncated), where the source and
    the wave launch live.  The physics loss is then a weighted residual integral; the weighting is declared."""
    if mode == 'uniform':
        return rng.uniform(0, XMAX, n), rng.uniform(1e-3, TMAX, n)
    n1 = n // 2; n2 = n - n1
    x1 = rng.uniform(0, XMAX, n1); t1 = rng.uniform(1e-3, TMAX, n1)
    x2 = np.minimum(rng.exponential(1.0, n2), XMAX); t2 = np.clip(rng.exponential(0.7, n2), 1e-3, TMAX)
    return np.concatenate([x1, x2]), np.concatenate([t1, t2])

def train(model, case, gamma, regime='physics', iters=3000, lr=5e-3, n_c=1024, n_b=128, n_d=256,
          seed=0, log_every=500, val_points=2000, verbose=True, sampling='mixed', weights=(1.0, 1.0, 1.0),
          w_scale=None, lbfgs_iters=0, front=False, param='sigma', ic_weight=1.0, ic_xmin=0.3):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    F = Fields(model, gamma, case, param=param, front=front)
    # fixed far-boundary data and validation/data sets from the exact solution
    tbb = rng.uniform(1e-3, TMAX, 4096); Tb_all, Sb_all = gt.exact(np.full_like(tbb, XMAX), tbb, case, gamma)
    xv = rng.uniform(0, XMAX, val_points); tv = rng.uniform(1e-3, TMAX, val_points); Tv, Sv = gt.exact(xv, tv, case, gamma)
    xd = rng.uniform(0, XMAX, n_d); td = rng.uniform(1e-3, TMAX, n_d); Td, Sd = gt.exact(xd, td, case, gamma)
    xd, td, Td, Sd = map(torch.as_tensor, (xd, td, Td, Sd))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    t0 = time.time(); hist = []; gnorms = []
    for it in range(1, iters + 1):
        opt.zero_grad()
        parts = {}
        if regime in ('physics', 'combined'):
            xc_np, tc_np = sample_colloc(rng, n_c, sampling); xc = torch.as_tensor(xc_np); tc = torch.as_tensor(tc_np)
            ib = rng.integers(0, 4096, n_b)
            tb = torch.as_tensor(tbb[ib]); Tb6 = torch.as_tensor(Tb_all[ib]); Sb6 = torch.as_tensor(Sb_all[ib])
            loss, parts = physics_loss(F, xc, tc, tb, Tb6, Sb6, weights, w_scale, ic_weight, ic_xmin)
        else:
            loss = torch.zeros(())
        if regime in ('data', 'combined'):
            ld = data_loss(F, xd, td, Td, Sd); parts['data'] = ld.item(); loss = loss + ld
        loss.backward()
        gn = math.sqrt(sum((p.grad ** 2).sum().item() for p in model.parameters() if p.grad is not None))
        gnorms.append(gn)
        opt.step(); sched.step()
        if it % log_every == 0 or it == iters:
            with torch.no_grad():
                T, S = F.sigma(torch.as_tensor(xv), torch.as_tensor(tv))
                val = gt.rel_l2(T.numpy(), Tv) + gt.rel_l2(S.numpy(), Sv)
            hist.append(dict(it=it, loss=loss.item(), val=val, **parts))
            if verbose: print("  it %5d loss %.3e val %.4f %s  (%.0fs)" % (it, loss.item(), val, parts, time.time() - t0), flush=True)
    if lbfgs_iters > 0 and regime in ('physics', 'combined'):
        xf_np, tf_np = sample_colloc(rng, 4 * n_c, sampling); xf = torch.as_tensor(xf_np); tf = torch.as_tensor(tf_np)
        ibf = rng.integers(0, 4096, 4 * n_b); tbf = torch.as_tensor(tbb[ibf]); Tbf = torch.as_tensor(Tb_all[ibf]); Sbf = torch.as_tensor(Sb_all[ibf])
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters, max_eval=int(lbfgs_iters * 1.25),
                               tolerance_grad=1e-12, tolerance_change=1e-14, history_size=50, line_search_fn='strong_wolfe')
        state = dict(n=0)
        def closure():
            lb.zero_grad()
            loss, _ = physics_loss(F, xf.clone(), tf.clone(), tbf, Tbf, Sbf, weights, w_scale, ic_weight, ic_xmin)
            if regime == 'combined': loss = loss + data_loss(F, xd, td, Td, Sd)
            loss.backward(); state['n'] += 1
            return loss
        lb.step(closure)
        with torch.no_grad():
            T, S = F.sigma(torch.as_tensor(xv), torch.as_tensor(tv))
            val = gt.rel_l2(T.numpy(), Tv) + gt.rel_l2(S.numpy(), Sv)
        hist.append(dict(it=iters, lbfgs_evals=state['n'], val=val))
        if verbose: print("  lbfgs %d evals  val %.4f  (%.0fs)" % (state['n'], val, time.time() - t0), flush=True)
    res = evaluate(F, gamma, case)
    res.update(dict(train_s=time.time() - t0, iters=iters, n_params=model.n_params(), val=hist[-1]['val'],
                    hist=hist, gradnorm_first=float(np.mean(gnorms[:50])), gradnorm_last=float(np.mean(gnorms[-50:]))))
    return res

def make_model(kind, n_q=6, L=4, target=None, seed=0, n_in=2):
    if kind == 'vqc': return VQC(n_q, L, seed=seed, n_in=n_in)
    if kind == 'mlp': return matched_mlp(target, seed=seed, n_in=n_in)
    if kind == 'ff': return matched_ff(target, seed=seed, n_in=n_in)
    if kind == 'mlpref': return MLP(64, 4, seed=seed, n_in=n_in)
    raise ValueError(kind)

# ----------------------------------------------------------------------------- E2: gradient variance at init
def grad_variance(n_q, L, loss_type, case='exp', gamma=9.0, n_init=200, n_pts=256, seed=0):
    rng = np.random.default_rng(seed)
    xc = torch.as_tensor(rng.uniform(0, XMAX, n_pts)); tc = torch.as_tensor(rng.uniform(1e-3, TMAX, n_pts))
    tb = torch.as_tensor(rng.uniform(1e-3, TMAX, 64)); Tb6, Sb6 = gt.exact(np.full(64, XMAX), tb.numpy(), case, gamma)
    Tb6, Sb6 = torch.as_tensor(Tb6), torch.as_tensor(Sb6)
    Td, Sd = gt.exact(xc.numpy(), tc.numpy(), case, gamma); Td, Sd = torch.as_tensor(Td), torch.as_tensor(Sd)
    gs = []
    for i in range(n_init):
        m = VQC(n_q, L, seed=10000 * seed + i); F = Fields(m, gamma, case)
        if loss_type == 'data': loss = data_loss(F, xc, tc, Td, Sd)
        elif loss_type == 'physics': loss, _ = physics_loss(F, xc.clone(), tc.clone(), tb, Tb6, Sb6)
        else:
            lp, _ = physics_loss(F, xc.clone(), tc.clone(), tb, Tb6, Sb6); loss = lp + data_loss(F, xc, tc, Td, Sd)
        g = torch.autograd.grad(loss, m.rot)[0][0, 0, 1].item()      # first-layer RY angle on qubit 0
        gs.append(g)
    gs = np.array(gs)
    return dict(n_q=n_q, L=L, loss=loss_type, var=float(gs.var()), mean_abs=float(np.abs(gs).mean()), n_init=n_init)

# ----------------------------------------------------------------------------- CLI
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['smoke', 'pilot', 'e1', 'e2', 'e3', 'run'])
    ap.add_argument('--model', default='vqc'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default='')
    ap.add_argument('--ic_weight', type=float, default=1.0); ap.add_argument('--ic_xmin', type=float, default=0.3); ap.add_argument('--n_b', type=int, default=128)
    ap.add_argument('--target', type=int, default=0, help='override the parameter-count target for matched classical nets (0 = VQC count)')
    ap.add_argument('--case', default='exp'); ap.add_argument('--gamma', type=float, default=9.0)
    ap.add_argument('--n_q', type=int, default=6); ap.add_argument('--L', type=int, default=4)
    ap.add_argument('--iters', type=int, default=3000); ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--regime', default='physics')
    ap.add_argument('--models', default='vqc,mlp,ff'); ap.add_argument('--out', default='results')
    ap.add_argument('--n_c', type=int, default=1024); ap.add_argument('--n_d', type=int, default=256)
    ap.add_argument('--sampling', default='mixed'); ap.add_argument('--lbfgs', type=int, default=0); ap.add_argument('--w_scale', type=float, default=None)
    a = ap.parse_args(); os.makedirs(os.path.join(HERE, a.out), exist_ok=True)
    if a.cmd == 'smoke':
        m = VQC(4, 2); F = Fields(m, a.gamma, a.case)
        x = torch.tensor([0.5, 1.0]); t = torch.tensor([0.5, 1.0])
        loss, parts = physics_loss(F, x, t, torch.tensor([0.3, 0.7]), torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
        loss.backward(); print("smoke: loss", loss.item(), parts, "grad ok", m.rot.grad.abs().sum().item() > 0)
        for k, tgt in [('mlp', m.n_params()), ('ff', m.n_params())]:
            mm = make_model(k, target=tgt); print(k, 'matched params', mm.n_params(), 'target', tgt)
        r = train(VQC(4, 2), a.case, a.gamma, iters=50, log_every=25); print({k: v for k, v in r.items() if k != 'hist'})
    elif a.cmd in ('pilot', 'e1'):
        vq = VQC(a.n_q, a.L); target = vq.n_params(); out = []
        for kind in a.models.split(','):
            for s in range(a.seeds):
                m = make_model(kind, a.n_q, a.L, target, seed=s)
                print("== %s seed %d params %d case %s gamma %g regime %s" % (kind, s, m.n_params(), a.case, a.gamma, a.regime), flush=True)
                r = train(m, a.case, a.gamma, regime=a.regime, iters=a.iters, lr=a.lr, n_c=a.n_c, n_d=a.n_d, seed=s, sampling=a.sampling, w_scale=a.w_scale, lbfgs_iters=a.lbfgs)
                r.update(dict(model=kind, seed=s, case=a.case, gamma=a.gamma, regime=a.regime, lr=a.lr, n_q=a.n_q, L=a.L))
                out.append(r); print("   -> relL2 T %.4f sigma %.4f (%.0fs)" % (r['relL2_T'], r['relL2_sigma'], r['train_s']), flush=True)
                json.dump(out, open(os.path.join(HERE, a.out, '%s_%s_g%g_%s_q%dL%d.json' % (a.cmd, a.case, a.gamma, a.regime, a.n_q, a.L)), 'w'), indent=1)
    elif a.cmd == 'run':
        vq = VQC(a.n_q, a.L); target = a.target if a.target > 0 else vq.n_params()
        m = make_model(a.model, a.n_q, a.L, target, seed=a.seed)
        name = '%s_%s_g%g_%s_q%dL%d_s%d%s' % (a.model, a.case, a.gamma, a.regime, a.n_q, a.L, a.seed, ('_' + a.tag) if a.tag else '')
        print("== run", name, "params", m.n_params(), flush=True)
        r = train(m, a.case, a.gamma, regime=a.regime, iters=a.iters, lr=a.lr, n_c=a.n_c, n_b=a.n_b, n_d=a.n_d, seed=a.seed,
                  sampling=a.sampling, w_scale=a.w_scale, lbfgs_iters=a.lbfgs, ic_weight=a.ic_weight, ic_xmin=a.ic_xmin, log_every=500)
        r.update(dict(model=a.model, seed=a.seed, case=a.case, gamma=a.gamma, regime=a.regime, lr=a.lr, n_q=a.n_q, L=a.L, n_c=a.n_c,
                      n_d=a.n_d, lbfgs=a.lbfgs, sampling=a.sampling, ic_weight=a.ic_weight, ic_xmin=a.ic_xmin, target_params=target, name=name))
        json.dump(r, open(os.path.join(HERE, a.out, name + '.json'), 'w'), indent=1)
        torch.save(m.state_dict(), os.path.join(HERE, a.out, name + '.pt'))
        print("   -> relL2 T %.4f sigma %.4f (%.0fs)" % (r['relL2_T'], r['relL2_sigma'], r['train_s']), flush=True)
    elif a.cmd == 'e2':
        out = []
        for L in (2, 4, 8):
            for n_q in range(2, 11):
                for lt in ('data', 'physics', 'combined'):
                    r = grad_variance(n_q, L, lt, a.case, a.gamma, n_init=200); out.append(r)
                    print(r, flush=True)
                    json.dump(out, open(os.path.join(HERE, a.out, 'e2_%s_g%g.json' % (a.case, a.gamma)), 'w'), indent=1)
