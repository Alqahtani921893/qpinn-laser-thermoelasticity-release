"""family_equivalence.py -- output-family equivalence of the depth-2 circuits (referee round 2, finding 2.2).

Claim tested: with Z readout on qubits 0 and 1 after a CZ ring, the last CZ layer is diagonal and cancels in the
Z expectations, and the preceding CZ ring couples the measured qubits only to their ring neighbours.  For L = 2 the
output therefore depends on the parameters of qubits {0, 1, 2, n_q - 1} only, so the 4-, 6- and 8-qubit depth-2
circuits realise the same function family up to the relabelling {0,1,2,3} <-> {0,1,2,n_q-1}.  The test transplants
the encoding scales, rotation angles and readout affine parameters of a 4-qubit circuit onto those qubits of a
6- and an 8-qubit circuit (other qubits random) and compares the outputs on 64 random inputs.  L = 4 is included
as a control: there the deeper light cone reaches every qubit and the outputs differ.
Run from analysis/:  OMP_NUM_THREADS=2 python3 family_equivalence.py
"""
import sys, torch
sys.path.insert(0, '.')
import qpinn_local as q

def transplant(src, n_q, map_q, seed):
    dst = q.VQC(n_q, src.L, seed=seed)
    with torch.no_grad():
        for s_q, d_q in enumerate(map_q):
            dst.enc[:, d_q, :] = src.enc[:, s_q, :]
            dst.rot[:, d_q, :] = src.rot[:, s_q, :]
        dst.out_a.copy_(src.out_a); dst.out_b.copy_(src.out_b)
    return dst

if __name__ == '__main__':
    torch.manual_seed(0)
    B = 64
    u = torch.rand(B, dtype=torch.float64) * 6; v = torch.rand(B, dtype=torch.float64) * 3
    print('family equivalence test: 64 random inputs (x in [0,6], t in [0,3]); float64/complex128 statevector')
    for L in (2, 4):
        src = q.VQC(4, L, seed=3)
        with torch.no_grad():
            src.out_a.copy_(torch.tensor([1.3, -0.7])); src.out_b.copy_(torch.tensor([0.2, 0.5]))
            y4 = src(u, v)
            for n_q in (6, 8):
                dst = transplant(src, n_q, [0, 1, 2, n_q - 1], seed=11)
                d = (dst(u, v) - y4).abs().max().item()
                print('L=%d: 4-qubit parameters transplanted onto qubits (0,1,2,%d) of the %d-qubit circuit: max |output difference| = %.2e %s'
                      % (L, n_q - 1, n_q, d, '(identical function family)' if d < 1e-12 else '(different: control)'))
