#!/bin/zsh
# Concatenate the analysis code into ../paper/results_package/code_bundle.txt (one text file for review and upload).
cd "$(dirname "$0")"
set -e
# ground_truth.py lives in ../chatgpt_pro_job/inputs/ in the working tree and next to this script in the release archive.
if [ -f ../chatgpt_pro_job/inputs/ground_truth.py ]; then GT=../chatgpt_pro_job/inputs/ground_truth.py; else GT=ground_truth.py; fi
mkdir -p ../paper/results_package
out=../paper/results_package/code_bundle.txt
{
  echo "# Code bundle for reproduction (QPINN-LTE-2026-10-20), regenerated $(date '+%Y-%m-%d %H:%M %Z'). Each file is delimited by '===== FILE: name ====='."
  echo "# ground_truth.py is the exact-solution module imported by qpinn_local.py (kept in chatgpt_pro_job/inputs/ in the working tree; next to the scripts in the release archive)."
  for f in qpinn_local.py $GT noise_eval.py run_lrsel.sh run_lrsel_hi.sh select_lrs.py run_e1.sh \
           run_functional_matched.sh functional_count.py family_equivalence.py run_ablation_xmin.sh run_e3.sh run_e3_aux.sh run_gamma.sh run_e4.py make_results_package.py \
           make_runs_all.py make_code_bundle.sh closed_form_corrected.py laplace_ref.py fd_ref.py check_corrected.py; do
    if [ ! -f "$f" ]; then echo "make_code_bundle.sh: missing input $f" >&2; exit 1; fi
    echo; echo "===== FILE: $(basename $f) ====="; cat "$f"
  done
} > $out
wc -c $out
