#!/usr/bin/env python3
"""make_runs_all.py -- collect every per-run record (results/*/*.json written by qpinn_local.py run) into
../paper/results_package/runs_all.json: a list of dicts with an added 'experiment' key; 'hist' (Adam checkpoints every
500 steps and the L-BFGS closure count) is kept.  Learning-rate selection runs are included (experiment lrsel*)."""
import json, glob, os, collections
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'paper', 'results_package')
SETS = [('e1', 'results/e1/*.json'), ('fmatch', 'results/fmatch/*.json'), ('ablation_xmin', 'results/ablation/*.json'), ('ablation_repro', 'results/ablation_repro/*.json'),
        ('e3', 'results/e3/*.json'), ('gamma', 'results/gamma/*.json'), ('lrsel', 'results/lrsel/*.json'),
        ('lrsel_fmatch', 'results/fmatch_lrsel/*.json'), ('lrsel_e3_data', 'results/e3_lrsel_data/*.json')]
rows = []
for exp, pat in SETS:
    for f in sorted(glob.glob(os.path.join(HERE, pat))):
        if os.path.basename(f).startswith('selected_lrs'): continue
        if os.environ.get('SKIP_PENDING') and exp in ('e3', 'gamma'): continue   # sweeps still running: not released yet
        r = json.load(open(f)); r['experiment'] = exp; r['file'] = os.path.relpath(f, HERE); rows.append(r)
json.dump(rows, open(os.path.join(OUT, 'runs_all.json'), 'w'))
print(len(rows), 'records:', dict(collections.Counter(r['experiment'] for r in rows)))
