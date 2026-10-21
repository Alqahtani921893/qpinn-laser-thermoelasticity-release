"""E4: evaluate trained VQC models (best two seeds per case/config from results/e1, plus results/gamma) under finite shots
and per-layer single-qubit depolarising noise.  Inference only; exact density-matrix simulation; 41 x 40 subgrid.
Writes results/e4_noise.json."""
import sys, os, json, glob, collections, torch, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qpinn_local as q, noise_eval as ne
cands = collections.defaultdict(list)
for pt in sorted(glob.glob('results/e1/vqc_*.pt') + glob.glob('results/gamma/vqc_*.pt')):
    meta = json.load(open(pt[:-3] + '.json')); cands[(meta['case'], meta['gamma'], meta['n_q'], meta['L'])].append((meta['relL2_sigma'], pt, meta))
rows = []
for key, lst in sorted(cands.items()):
    for _, pt, meta in sorted(lst, key=lambda z: z[0])[:2]:
        m = q.VQC(meta['n_q'], meta['L']); m.load_state_dict(torch.load(pt)); m.eval()
        base = dict(name=meta['name'], case=meta['case'], gamma=meta['gamma'], seed=meta['seed'], n_q=meta['n_q'], L=meta['L'],
                    clean_T=meta['relL2_T'], clean_sigma=meta['relL2_sigma'])
        for p in (0.0, 1e-3, 1e-2):
            for shots in (None, 1000, 10000, 100000):
                pass
                r = ne.evaluate_noisy(m, meta['gamma'], meta['case'], p=p, shots=shots, n=41, seed=meta['seed'])
                rows.append(dict(base, **r)); print(meta['name'], p, shots, "T %.4f sigma %.4f" % (r['relL2_T'], r['relL2_sigma']), flush=True)
        json.dump(rows, open('results/e4_noise.json', 'w'), indent=1)
print("E4_DONE", len(rows))
