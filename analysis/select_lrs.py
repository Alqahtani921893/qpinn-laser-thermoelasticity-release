"""Pick the learning rate per (case, model, n_q, L) by validation error from results/lrsel; write results/lrsel/selected_lrs.json."""
import json, glob, collections, sys
rows = collections.defaultdict(dict)
for f in [p for p in glob.glob('results/lrsel/*.json') if not p.endswith('selected_lrs.json')]:
    r = json.load(open(f)); rows[(r['case'], r['model'], r['n_q'], r['L'])][r['lr']] = r['val']
sel = {}
for key in sorted(rows):
    best = min(rows[key], key=lambda lr: rows[key][lr]); sel['%s|%s|%d|%d' % key] = best
    print("%-6s %-6s q%dL%d  best lr %-6g  " % (key[0], key[1], key[2], key[3], best) + "  ".join("lr%g: %.3f" % (lr, rows[key][lr]) for lr in sorted(rows[key])))
json.dump(sel, open('results/lrsel/selected_lrs.json', 'w'), indent=1); print(len(sel), "selections written")
