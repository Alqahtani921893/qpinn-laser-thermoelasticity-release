#!/bin/zsh
# Auxiliary E3 dispatcher (v2, 21 Oct 2026 07:20; a first v2 start at 07:08 used pgrep -c, which macOS lacks, and dispatched
# 13 jobs at once; they were killed and their logs set aside as logs/e3/aborted_*.log). History: the main run_e3.sh queue (3 workers, file order) was stopped at
# 06:35 because the two N_c = 4096 circuit jobs would have run last; v1 of this script (xargs, 5 workers, longest first)
# started them first, but at 07:10 they were dropped from the sweep (about 9 h per seed unloaded: Adam 2000 on 4096 points
# plus L-BFGS on 16384 points) and killing them made BSD xargs abort its queue.  v2 is a plain loop: it runs every job line
# of logs/e3/jobs.txt whose log file does not exist (= not started), longest first, keeping the number of E3 circuit
# processes (including jobs started earlier that are still running) at the value in logs/e3/aux_workers (default 4,
# editable while running), then waits for every E3 process to end and stamps E3_DONE into logs/e3_run.out.
# Job lines, learning rates and outputs are exactly those of logs/e3/jobs.txt (written by run_e3.sh).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
jobs_file=logs/e3/jobs.txt; aux=logs/e3/jobs_aux.txt; wfile=logs/e3/aux_workers
[[ -f $wfile ]] || echo 4 > $wfile
python3 - "$jobs_file" "$aux" <<'PY'
import re, os, sys
src, dst = sys.argv[1], sys.argv[2]
def cost(line):
    model = re.search(r'--model (\w+)', line).group(1)
    nq = int(re.search(r'--n_q (\d+)', line).group(1)); L = int(re.search(r'--L (\d+)', line).group(1))
    nc = re.search(r'--n_c (\d+)', line); nc = int(nc.group(1)) if nc else 0
    if model != 'vqc': return 1
    if '--regime data' in line: return 10
    return (2 ** nq) * L * (nc + 100) / 1000
rows = []
for line in open(src):
    line = line.rstrip('\n')
    if not line: continue
    log = re.search(r'> (logs/e3/\S+\.log)', line).group(1)
    if os.path.exists(log): continue
    rows.append((cost(line), line))
rows.sort(key=lambda r: -r[0])
open(dst, 'w').write('\n'.join(r[1] for r in rows) + '\n')
print('aux jobs:', len(rows))
for c, l in rows: print('%7.1f  %s' % (c, l.split('run ')[1].split(' --out')[0]))
PY
pat='[Pp]ython[^ ]* qpinn_local.py run .*--out results/e3'
jobs=("${(@f)$(cat $aux)}")
i=1
while (( i <= ${#jobs} )); do
  n=$(cat $wfile 2>/dev/null || echo 4); running=$(pgrep -f "$pat" | wc -l | tr -d " ")
  if (( running < n )); then
    echo "$(date '+%H:%M') start [$i/${#jobs}] ${jobs[i]%% > *}" | sed 's/python3 qpinn_local.py run //'
    zsh -c "${jobs[i]}" &
    (( i++ )); sleep 3
  else
    sleep 30
  fi
done
wait
while pgrep -f "$pat" > /dev/null; do sleep 60; done
echo "E3 aux finished $(date): $(ls results/e3/*.json | wc -l | tr -d ' ') of $(grep -c . $jobs_file) results (the two VQC N_c = 4096 jobs were dropped)" >> logs/e3_run.out
echo E3_DONE >> logs/e3_run.out
