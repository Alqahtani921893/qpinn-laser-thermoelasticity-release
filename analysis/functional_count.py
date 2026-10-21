#!/usr/bin/env python3
"""functional_count.py -- coordinate-perturbation audit of the circuit parameters (which parameters can affect the outputs).
Every trainable scalar of a freshly initialised VQC (seed 0) is perturbed by an additive +0.37 in its native units (radians for rotation angles) and the two outputs are re-evaluated
on 64 fixed random inputs (u, v ~ U(-1, 1)); a coordinate is 'inert' when max |delta y| < 1e-12.  Reports the nominal count,
the inert count split into encoding scales and rotation angles, the functional (output-affecting) count, and the layer/qubit
location of every inert coordinate.  Usage: python3 functional_count.py [n_q L ...]   (default: the six E3 configurations)."""
import sys, torch, qpinn_local as q
torch.set_default_dtype(torch.float64)
cfgs = [(4, 2), (4, 4), (6, 2), (6, 4), (6, 8), (8, 2)]
if len(sys.argv) > 1: cfgs = [(int(sys.argv[i]), int(sys.argv[i + 1])) for i in range(1, len(sys.argv), 2)]
g = torch.Generator().manual_seed(12345); u = torch.rand(64, generator=g) * 2 - 1; v = torch.rand(64, generator=g) * 2 - 1
for n_q, L in cfgs:
    m = q.VQC(n_q, L, seed=0)
    with torch.no_grad():
        y0 = m(u, v).clone()
        inert, nominal = [], 0
        for name, p in m.named_parameters():
            flat = p.view(-1)
            for i in range(flat.numel()):
                nominal += 1; old = flat[i].item(); flat[i] = old + 0.37
                d = (m(u, v) - y0).abs().max().item(); flat[i] = old
                if d < 1e-12: inert.append((name, i))
    enc = [(n, i) for n, i in inert if n == 'enc']; rot = [(n, i) for n, i in inert if n == 'rot']
    def loc(name, i):
        if name == 'enc': l, r = divmod(i, n_q * 2); qb, k = divmod(r, 2); return 'L%d q%d %s' % (l, qb, 'a' if k == 0 else 'b')
        if name == 'rot': l, r = divmod(i, n_q * 3); qb, k = divmod(r, 3); return 'L%d q%d theta%d' % (l, qb, k + 1)
        return name
    print('%dq%dL: nominal %d, inert %d (encoding %d, rotation %d), functional %d' % (n_q, L, nominal, len(inert), len(enc), len(rot), nominal - len(inert)))
    print('   inert:', ', '.join(loc(n, i) for n, i in inert))
