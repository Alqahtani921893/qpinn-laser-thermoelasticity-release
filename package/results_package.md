# Results package: what was run and how (authoritative description for the manuscript)

Package v6 (21 Oct 2026, 11:45; wording revised 12:20 per the writer's round-2 open issue 2: perturbation units, thread-reproducibility sentence): the v5 tables (unchanged numbers) for every experiment: E1, E1b (functional-matched nets),
the x_min and the initial-rate ablations, E2, E3 (66 runs: Pareto, collocation and labelled-point sweeps), E4 (corrected, with
clean same-grid controls, grouped by wave speed) and the secondary wave speeds (24 runs). Changes from v5, all textual, after
referee round 2 (finding 1.2, consistency of the ancillary material with the revised manuscript): the initial-rate sentence
now quotes the executed single-seed ablation values instead of a general "26 percent plateau"; the Fourier-feature minimum
of 40 parameters is described as the smallest searched width (19h + 2 gives 21 at h = 1, not searched); the E1-versus-
functional-count deviations are stated per net (+40.0, +33.3, +29.5, +31.8 percent); the field-figure caption no longer
infers a representational limit from the physics-trained circuit; the E2 caption quotes the measured multipliers and fitted
slopes instead of "decay unchanged"; the finite-difference maxima are given per pulse (T 4.20e-6 / sigma 6.18e-3 exponential,
T 5.91e-6 / sigma 6.19e-3 rectangular) in closed_form_appendix.md and in the ground_truth.py header, which now states the
scope of each verification. New executed check (referee finding 2.2): family_equivalence.py / family_equivalence_check.txt
shows that the depth-2 circuits with 4, 6 and 8 qubits realise the same output family (4-qubit parameters transplanted onto
qubits 0, 1, 2 and n_q - 1 reproduce the outputs to 1e-15; depth 4 differs), so the E3 rows (4,2), (6,2), (8,2) are three
differently initialised realisations of one 30-coordinate family with different simulation cost. The field figure
fig_fields_exp.pdf was regenerated at its final placed size (6.6 x 3.7 in, 7-8 pt labels) for legibility (finding 5.1).

Everything below was executed on one Apple M2 Max (12 cores, CPU only), Python 3.12.5, PyTorch 2.9.1, NumPy 2.5.1,
SciPy 1.16.3, float64/complex128, exact statevector simulation (analysis/qpinn_local.py).  Exact fields come from
ground_truth.py (corrected closed forms, verified against de Hoog inverse Laplace and finite differences).

## Problem and benchmark (unchanged from problem_spec.md)
Nondimensional heat equation with Beer-Lambert source and one-way coupled stress wave; x in [0, 6], t in [0, 3];
gamma = 9 (secondary 25, 49); pulses exp (beta = 2, Ae = 1), step (As = 1, Delta = 0.5), smooth (beta1 = 2, beta2 = 8,
A = 4/3), all with pulse energy 0.5.  Test grid: 241 x-nodes on [0, 6] times 240 t-nodes on (0, 3]; metric: relative
L2 error of T and of sigma separately, plus max abs error; five seeds (0-4) per cell; mean +- sample sd reported.

## Models (two inputs u = x/3 - 1, v = 2t/3 - 1; two outputs y0, y1)
- QPINN (VQC): data re-uploading circuit, n_q qubits, L layers.  Per layer and qubit the fused single-qubit gate is
  RZ(theta3) RY(theta2) RZ(theta1) RX(b v) RY(a u) with trainable encoding scales a, b ~ N(0,1) at init and trainable
  rotation angles theta ~ U(0, 2 pi); then a CZ ring (q, q+1 mod n_q).  Outputs y_k = alpha_k <Z_k> + beta_k, k = 0, 1,
  with trainable alpha (init 1) and beta (init 0).  Parameters: 5 n_q L + 4 (4q2L: 44; 6q4L: 124).  Exact statevector,
  derivatives by autograd (PyTorch complex128), third-order graphs for the second derivatives in the residuals.
- MLP-matched: tanh MLP with two hidden layers, width chosen so that the parameter count is closest to the VQC's
  (42 vs 44; 114 vs 124).
- FF-matched: eight fixed random Fourier features (frequencies ~ N(0, pi^2)) -> cos/sin (16 features) -> one tanh hidden
  layer -> linear; width chosen to match (40 vs 44; 116 vs 124).  Dequantisation comparator.
- MLP-ref: tanh MLP 2 -> 64 -> 64 -> 64 -> 64 -> 2, 12,802 parameters (attainable-accuracy reference).

## Hard constraints, losses, training (identical for every family)
- Hard initial conditions: T = t y0, sigma = t y1 (both vanish at t = 0).
- Physics loss on N_c = 512 collocation points per step, resampled every step: half uniform on the domain, half
  concentrated near the heated corner (x ~ Exp(1) clipped to [0, 6], t ~ Exp(0.7) clipped to [1e-3, 3]).  Residuals
  r_h = T_t - T_xx - e^{-x} f(t) and r_s = sigma_xx - sigma_tt/gamma - T_tt, mean squares, weight 1 each.
- Boundary terms (128 points per step, weight 1): T_x(0,t) = 0; sigma(0,t) = 0; at x = 6 exact Dirichlet data
  T(6,t), sigma(6,t) from ground_truth.py (domain truncation of the semi-infinite problem; declared).
- Initial-rate terms (weight 1, points x >= 0.3 of the collocation sample at t = 0): y0(x,0) = e^{-x} f(0+) and
  (y1(x,0) + gamma e^{-x} f(0+))/gamma = 0.  Without the stress rate the stress problem is under-determined (the PDE and
  boundary data do not fix sigma_t(x,0+)); in the single-seed weight ablation (ablation_initial_rate.md) removing the term
  raised the stress error of the reference MLP from 0.0492 to 0.2152 and of the matched MLP from 0.1827 to 0.8132 (no general
  plateau is claimed); the rates are discontinuous at the free-surface corner, hence the x >= 0.3 restriction.
- Optimiser: Adam, 2000 steps, cosine schedule to 5 percent of the initial rate, then L-BFGS (strong Wolfe, history 50)
  for up to 1000 iterations on a fixed 2048-point collocation set (same sampling) and 512 boundary points.
- Learning rate chosen per (case, family, configuration) by validation on 2000 random points (never the test grid)
  from {1e-3, 3e-3, 1e-2, 3e-2, 1e-1} with Adam 1000 steps and seed 100 (results/lrsel/selected_lrs.json): VQC 0.1
  (exp both configs; smooth 4q2L), 0.03 (step both; smooth 6q4L 0.01); MLP-matched 0.1/0.03/0.001; FF 0.1/0.03/0.01;
  MLP-ref 0.01 (all cases).
- Regimes: physics-only (E1, E3 Pareto and collocation sweep), data-only (E3 labelled-point sweep: MSE on N_d exact
  samples, no PDE loss, Adam 2000 only).

## Experiments
- E1: 3 cases x 5 seeds x {VQC 4q2L, VQC 6q4L, MLP-matched x2, FF-matched x2, MLP-ref} = 105 runs (results/e1).
- E1b (functional-matched): classical nets matched to the circuits' FUNCTIONAL (output-affecting) parameter counts, 30 for
  4q2L and 88 for 6q4L (numerical perturbation test, functional_parameter_counts.md): MLP 29 and 93, FF 40 (the smallest searched width, h = 2;
  19h + 2 gives 21 at h = 1, which was not searched; this row equals the E1 network and is not a valid match) and 97; learning rates re-selected per cell with
  the E1 procedure (results/fmatch_lrsel); 3 cases x 5 seeds x 4 nets = 60 runs (results/fmatch; run_functional_matched.sh).
- Ablations (seed 0, exp pulse, gamma 9): (a) initial-rate weight in {0, 0.1, 1} for MLP-ref and MLP-matched (N_c 1024,
  lr 3e-3; log-derived, ablation_initial_rate.md); (b) near-surface exclusion width x_min in {0, 0.1, 0.3, 0.6, 1.0} for
  MLP-ref (lr 0.01), extended to the 6q4L circuit and its matched MLP at x_min in {0, 0.1} (lr 0.1; their x_min = 0.3 rows
  are the E1 seed-0 runs) (results/ablation; run_ablation_xmin.sh).  Reproducibility note: the MLP-ref x_min = 0.3 ablation run
  (2 threads) has a bitwise-identical Adam history to the E1 seed-0 run (3 threads) but a different L-BFGS endpoint (sigma
  0.0496 after 1060 closures vs 0.0461 after 1065); rerunning it with 3 threads reproduces the E1 run exactly
  (results/ablation_repro). A plausible explanation is that the intra-op thread count changes the floating-point
  reduction order and the strong-Wolfe line search amplifies the last-bit difference; it was not separately instrumented.
  The E3 reruns at 2 threads reproduced the E1 records (3 threads) bitwise for the circuits and the matched nets, so the
  thread dependence is not universal; these runs establish reproducibility of the reported repeats, not universal bitwise
  determinism.
- Functional (output-affecting) parameter audit: functional_count.py (seed 0, 64 random inputs, an additive perturbation of +0.37 in each coordinate's native units (radians for rotation angles),
  inert if max |delta y| < 1e-12); executed output in functional_parameter_audit.txt: 44 -> 30, 84 -> 70, 64 -> 30, 124 -> 88,
  244 -> 208, 84 -> 30, with the layer/qubit location of every inert coordinate.
- E2: variance over 200 random initialisations of dL/dtheta for the first-layer RY angle on qubit 0, n_q = 2..10,
  L in {2, 4, 8}, losses data (MSE on 256 exact samples), physics (full physics loss on the same 256 points + 64
  boundary points), combined; exp pulse, gamma 9 (results/e2_exp_g9.json).  Slope of log10 Var versus n_q fitted by
  least squares.
- E3: (i) Pareto: VQC configs (4,2), (4,4), (6,2), (6,4), (6,8), (8,2) and their matched MLP/FF, exp, 2 seeds;
  (ii) collocation sweep N_c in {64, 256, 1024, 4096} for VQC 6q4L and MLP-matched, physics-only, 2 seeds, as planned; executed:
  the circuit at N_c in {64, 256, 1024} (512 = its Pareto/E1 runs) and the MLP at all four values. The circuit at N_c = 4096 was
  started and stopped after 35 min: at the measured rates (Adam on 4096 points, then L-BFGS on 16,384 fixed points) each seed
  projected to about nine hours, so the two jobs were dropped (declared deviation; 66 of the planned 68 E3 runs);
  (iii) labelled-point sweep N_d in {16, 64, 256, 1024}, data-only, same models, 2 seeds (results/e3).
  Learning rates: the E1-selected exp-pulse rates (0.1 for every VQC, MLP-matched and FF-matched cell; MLP-ref 0.01); the
  data-only regime has its own validation selection with the E1 procedure (N_d 256): VQC 0.03, MLP 0.1 (results/e3_lrsel_data).
  Matched nets are nearest-count matches (the width whose count is closest to the circuit's nominal count); the deviation is
  reported per configuration with the E3 table and exceeds 10 percent for the targets 64 and 84.  A first launch of E3 and of the
  secondary wave speeds with hard-coded rates (5e-3 / 3e-3) was discarded before any result was used (results/*_superseded_wronglr);
  the sweeps were relaunched on 21 Oct 04:05 with the rates above (run_e3.sh; from 06:35 the remaining job lines were served
  by run_e3_aux.sh, longest first, with an adjustable worker count; a job is skipped when its log exists, and completion was
  checked from the per-cell records).
- E4: trained VQCs (best two seeds per case/config) evaluated on a 41 x 40 subgrid with (a) finite shots
  N in {1e3, 1e4, 1e5} (binomial sampling of each <Z_k>), (b) single-qubit depolarising noise p in {1e-3, 1e-2}
  after every layer on every qubit, exact density-matrix simulation (results/e4_noise.json).
- Secondary wave speeds: gamma in {25, 49}, exp pulse, VQC 6q4L, MLP-matched, FF-matched, MLP-ref, 3 seeds
  (results/gamma), same recipe and the gamma-9 exp-pulse learning rates (0.1; MLP-ref 0.01).

## Run commands
python3 qpinn_local.py run --model {vqc|mlp|ff|mlpref} --case {exp|step|smooth} --gamma 9 --n_q N --L L --seed S
    --iters 2000 --lbfgs 1000 --n_c 512 --lr LR --out results/e1          (see run_e1.sh; lrs from select_lrs.py)
python3 qpinn_local.py e2 --case exp --gamma 9                                (E2)
run_lrsel.sh + run_lrsel_hi.sh + select_lrs.py (learning-rate selection), run_e1.sh, run_functional_matched.sh (E1b, own
selection), run_ablation_xmin.sh, run_e3.sh (+ run_e3_aux.sh), run_gamma.sh, run_e4.py (E4), make_results_package.py (tables +
figures), make_runs_all.py (runs_all.json: every per-run record incl. the selection runs, Adam history every 500 steps and the
L-BFGS closure count), make_code_bundle.sh, functional_count.py.  Timing: wall-clock seconds per run (train_s), covering
training, validation and the final test-grid evaluation, on a shared machine.  Thread settings (OMP_NUM_THREADS) per launcher:
E1, learning-rate selection, E3 main queue and secondary wave speeds 3; E1b (functional-matched), the x_min ablation and the
auxiliary E3 queue 2; the E4 evaluator 4.  Concurrency: 3 to 4 processes for E1/E3/gamma, 3 for E1b, up to 11 circuit jobs
during the overlap of the E3, gamma and ablation queues on 21 Oct 06:35-08:00, when each process received 50-90 percent of a
core (train_s of those runs is inflated accordingly and is not comparable across sweeps).

## Note for E4 (readout amplification)
Trained output gains alpha_k of the E1 circuits (results/e1/vqc_*.pt): 6-qubit, 4-layer circuits end with alpha_T = 1.3-4.5
and |alpha_sigma| = 4.8-15.6; the 4-qubit, 2-layer circuits with |alpha| up to 836 (median about 50). A finite-shot
estimate of <Z_k> with N shots has standard deviation about 1/sqrt(N), which the gain multiplies, so the E4 shot-noise
errors (see the E4 table: ratios to the clean same-grid control of 1.6-22 at 1e3 shots and 1.0-2.6 at 1e5 shots) are a direct consequence of the gains the optimiser chose, not of
sampling alone. The manuscript should present this as a hardware-realism limitation of unconstrained output scaling.


# Results package (auto-generated by make_results_package.py)

## E1: parity at matched parameter count (physics-only, gamma = 9, five seeds)

| case | model | config | params | lr | T relL2 mean +- sd | sigma relL2 mean +- sd | best-seed sigma | train s |
|---|---|---|---|---|---|---|---|---|
| exp | FF-matched | q4L2 | 40 | 0.1 | 0.4500 +- 0.1532 | 0.7826 +- 0.2417 | 0.3669 | 45 |
| exp | MLP-matched | q4L2 | 42 | 0.1 | 0.3178 +- 0.0738 | 0.3055 +- 0.0608 | 0.2311 | 16 |
| exp | QPINN (VQC) | q4L2 | 44 | 0.1 | 0.4699 +- 0.1428 | 1.1137 +- 0.1711 | 0.8460 | 725 |
| exp | FF-matched | q6L4 | 116 | 0.1 | 0.2401 +- 0.1689 | 0.5719 +- 0.2035 | 0.3278 | 44 |
| exp | MLP-matched | q6L4 | 114 | 0.1 | 0.0990 +- 0.0584 | 0.2146 +- 0.1375 | 0.1055 | 20 |
| exp | MLP-ref | - | 12802 | 0.01 | 0.0125 +- 0.0029 | 0.0539 +- 0.0076 | 0.0461 | 215 |
| exp | QPINN (VQC) | q6L4 | 124 | 0.1 | 0.1779 +- 0.0439 | 0.7608 +- 0.1884 | 0.5698 | 4189 |
| smooth | FF-matched | q4L2 | 40 | 0.1 | 0.6403 +- 0.1968 | 0.9434 +- 0.1900 | 0.7787 | 43 |
| smooth | MLP-matched | q4L2 | 42 | 0.001 | 0.2220 +- 0.0870 | 0.5496 +- 0.3482 | 0.3627 | 15 |
| smooth | QPINN (VQC) | q4L2 | 44 | 0.1 | 0.3260 +- 0.0582 | 0.9826 +- 0.0958 | 0.8986 | 758 |
| smooth | FF-matched | q6L4 | 116 | 0.01 | 0.2524 +- 0.1288 | 0.6254 +- 0.2388 | 0.3706 | 43 |
| smooth | MLP-matched | q6L4 | 114 | 0.1 | 0.0519 +- 0.0143 | 0.1704 +- 0.0632 | 0.1172 | 20 |
| smooth | MLP-ref | - | 12802 | 0.01 | 0.0058 +- 0.0013 | 0.0296 +- 0.0106 | 0.0185 | 193 |
| smooth | QPINN (VQC) | q6L4 | 124 | 0.01 | 0.1664 +- 0.0342 | 0.5733 +- 0.0696 | 0.4760 | 4474 |
| step | FF-matched | q4L2 | 40 | 0.1 | 0.7012 +- 0.0570 | 0.9526 +- 0.0595 | 0.8714 | 42 |
| step | MLP-matched | q4L2 | 42 | 0.1 | 0.3314 +- 0.0683 | 0.4762 +- 0.0743 | 0.4254 | 15 |
| step | QPINN (VQC) | q4L2 | 44 | 0.03 | 0.6688 +- 0.1857 | 1.1934 +- 0.3074 | 0.8599 | 625 |
| step | FF-matched | q6L4 | 116 | 0.03 | 0.3405 +- 0.1200 | 0.3825 +- 0.0742 | 0.3289 | 47 |
| step | MLP-matched | q6L4 | 114 | 0.03 | 0.2105 +- 0.0904 | 0.3564 +- 0.1015 | 0.2300 | 21 |
| step | MLP-ref | - | 12802 | 0.01 | 0.0221 +- 0.0147 | 0.0517 +- 0.0086 | 0.0439 | 213 |
| step | QPINN (VQC) | q6L4 | 124 | 0.03 | 0.2579 +- 0.0336 | 0.8954 +- 0.1136 | 0.7391 | 3644 |

## E3: efficiency (exp pulse, gamma = 9, seeds 0 and 1; nominal = registered parameter count, functional = output-affecting count of the circuit; classical nets are nearest-count matches to the circuit's nominal count, deviation given in percent; lr 0.1 for every physics-only cell)

### Error versus parameter count (physics-only, N_c = 512 per Adam step, Adam 2000 + L-BFGS 1000 on 2048 fixed points)
| model | config | nominal params | functional / deviation from target | T relL2 mean | sigma relL2 mean | T per seed | sigma per seed | train s mean |
|---|---|---|---|---|---|---|---|---|
| FF-matched | q4L2 | 40 | -9.1% vs 44 | 0.4056 | 0.8616 | 0.4112, 0.4001 | 0.9487, 0.7746 | 33 |
| MLP-matched | q4L2 | 42 | -4.5% vs 44 | 0.2755 | 0.3067 | 0.2723, 0.2788 | 0.3823, 0.2311 | 13 |
| QPINN (VQC) | q4L2 | 44 | 30 functional | 0.4030 | 1.0127 | 0.2561, 0.5500 | 0.8460, 1.1794 | 837 |
| FF-matched | q4L4 | 78 | -7.1% vs 84 | 0.2830 | 0.5552 | 0.3864, 0.1796 | 0.6899, 0.4204 | 31 |
| MLP-matched | q4L4 | 93 | +10.7% vs 84 | 0.1052 | 0.2403 | 0.1100, 0.1005 | 0.2196, 0.2610 | 17 |
| QPINN (VQC) | q4L4 | 84 | 70 functional | 0.1947 | 0.9752 | 0.1905, 0.1990 | 1.0473, 0.9030 | 1383 |
| FF-matched | q6L2 | 59 | -7.8% vs 64 | 0.3909 | 0.8662 | 0.4830, 0.2988 | 0.8510, 0.8815 | 30 |
| MLP-matched | q6L2 | 57 | -10.9% vs 64 | 0.1530 | 0.3598 | 0.1020, 0.2040 | 0.2854, 0.4341 | 15 |
| QPINN (VQC) | q6L2 | 64 | 30 functional | 0.6330 | 1.4455 | 0.6498, 0.6162 | 1.3497, 1.5413 | 2848 |
| FF-matched | q6L4 | 116 | -6.5% vs 124 | 0.2159 | 0.7825 | 0.1650, 0.2667 | 0.7594, 0.8056 | 30 |
| MLP-matched | q6L4 | 114 | -8.1% vs 124 | 0.1402 | 0.3640 | 0.1959, 0.0844 | 0.3409, 0.3872 | 15 |
| QPINN (VQC) | q6L4 | 124 | 88 functional | 0.1657 | 0.7352 | 0.1939, 0.1376 | 0.8785, 0.5919 | 4399 |
| FF-matched | q6L8 | 249 | +2.0% vs 244 | 0.1420 | 0.3823 | 0.1327, 0.1513 | 0.3583, 0.4064 | 32 |
| MLP-matched | q6L8 | 249 | +2.0% vs 244 | 0.0557 | 0.1195 | 0.0359, 0.0756 | 0.0935, 0.1455 | 21 |
| QPINN (VQC) | q6L8 | 244 | 208 functional | 0.0598 | 0.4464 | 0.0606, 0.0591 | 0.3951, 0.4977 | 10117 |
| FF-matched | q8L2 | 78 | -7.1% vs 84 | 0.2830 | 0.5552 | 0.3864, 0.1796 | 0.6899, 0.4204 | 28 |
| MLP-matched | q8L2 | 93 | +10.7% vs 84 | 0.1052 | 0.2403 | 0.1100, 0.1005 | 0.2196, 0.2610 | 16 |
| QPINN (VQC) | q8L2 | 84 | 30 functional | 0.8757 | 2.0390 | 0.6694, 1.0819 | 1.7132, 2.3649 | 10070 |

### Error versus collocation points per Adam step (physics-only, VQC 6q4L and its nominal-matched MLP; L-BFGS 1000 on a fixed set of 4 N_c points; Adam 2000; lr 0.1)
| model | NC | T relL2 mean | sigma relL2 mean | T per seed | sigma per seed | train s mean |
|---|---|---|---|---|---|---|
| QPINN (VQC) | 64 | 0.2199 | 0.8198 | 0.1051, 0.3348 | 0.8078, 0.8318 | 1182 |
| QPINN (VQC) | 256 | 0.2158 | 0.6687 | 0.2140, 0.2176 | 0.5376, 0.7998 | 3152 |
| QPINN (VQC) | 512 | 0.1657 | 0.7352 | 0.1939, 0.1376 | 0.8785, 0.5919 | 4399 |
| QPINN (VQC) | 1024 | 0.2139 | 0.7594 | 0.1815, 0.2463 | 0.8250, 0.6938 | 9813 |
| MLP-matched | 64 | 0.1287 | 0.2041 | 0.0870, 0.1705 | 0.2023, 0.2060 | 9 |
| MLP-matched | 256 | 0.1198 | 0.2575 | 0.0906, 0.1491 | 0.2438, 0.2713 | 11 |
| MLP-matched | 512 | 0.1402 | 0.3640 | 0.1959, 0.0844 | 0.3409, 0.3872 | 15 |
| MLP-matched | 1024 | 0.0962 | 0.1726 | 0.0770, 0.1154 | 0.0886, 0.2566 | 21 |
| MLP-matched | 4096 | 0.1165 | 0.2968 | 0.1118, 0.1213 | 0.3441, 0.2496 | 65 |

### Error versus labelled exact samples (data-only) (MSE on N_d exact samples drawn once per run, no PDE loss, Adam 2000 only, no L-BFGS; lr 0.03 (VQC), 0.1 (MLP), both selected by validation in this regime)
| model | ND | T relL2 mean | sigma relL2 mean | T per seed | sigma per seed | train s mean |
|---|---|---|---|---|---|---|
| QPINN (VQC) | 16 | 1.6397 | 2.1382 | 0.9935, 2.2859 | 2.5130, 1.7633 | 28 |
| QPINN (VQC) | 64 | 0.2798 | 0.5336 | 0.2571, 0.3024 | 0.6260, 0.4412 | 32 |
| QPINN (VQC) | 256 | 0.1907 | 0.2918 | 0.1873, 0.1940 | 0.2953, 0.2882 | 48 |
| QPINN (VQC) | 1024 | 0.1920 | 0.2178 | 0.1769, 0.2070 | 0.2211, 0.2146 | 149 |
| MLP-matched | 16 | 1.0870 | 1.5074 | 0.7236, 1.4504 | 1.7778, 1.2370 | 1 |
| MLP-matched | 64 | 0.0802 | 0.1903 | 0.0865, 0.0738 | 0.2229, 0.1577 | 1 |
| MLP-matched | 256 | 0.1007 | 0.0884 | 0.0873, 0.1140 | 0.0773, 0.0995 | 1 |
| MLP-matched | 1024 | 0.0776 | 0.0872 | 0.0870, 0.0682 | 0.0885, 0.0858 | 1 |

## E1b: classical nets matched to the circuits' FUNCTIONAL parameter counts (30 for 4q2L, 88 for 6q4L; learning rates selected per cell by validation; five seeds; same recipe as E1)

| case | model | matched to | params | lr | T relL2 mean +- sd | sigma relL2 mean +- sd | best-seed sigma |
|---|---|---|---|---|---|---|---|
| exp | FF-matched | q4L2 functional (30) | 40 | 0.1 | 0.4500 +- 0.1532 | 0.7826 +- 0.2417 | 0.3669 |
| exp | MLP-matched | q4L2 functional (30) | 29 | 0.1 | 0.3322 +- 0.0579 | 0.4396 +- 0.1554 | 0.3135 |
| exp | FF-matched | q6L4 functional (88) | 97 | 0.1 | 0.3162 +- 0.1474 | 0.6332 +- 0.0497 | 0.6043 |
| exp | MLP-matched | q6L4 functional (88) | 93 | 0.1 | 0.1379 +- 0.0302 | 0.2857 +- 0.0607 | 0.2196 |
| smooth | FF-matched | q4L2 functional (30) | 40 | 0.1 | 0.6403 +- 0.1968 | 0.9434 +- 0.1900 | 0.7787 |
| smooth | MLP-matched | q4L2 functional (30) | 29 | 0.1 | 0.2652 +- 0.0251 | 0.4884 +- 0.0163 | 0.4676 |
| smooth | FF-matched | q6L4 functional (88) | 97 | 0.1 | 0.3688 +- 0.1425 | 0.8194 +- 0.1841 | 0.5460 |
| smooth | MLP-matched | q6L4 functional (88) | 93 | 0.1 | 0.0902 +- 0.0258 | 0.3631 +- 0.2001 | 0.1427 |
| step | FF-matched | q4L2 functional (30) | 40 | 0.1 | 0.7012 +- 0.0570 | 0.9526 +- 0.0595 | 0.8714 |
| step | MLP-matched | q4L2 functional (30) | 29 | 0.03 | 0.5283 +- 0.1594 | 0.7658 +- 0.3048 | 0.4344 |
| step | FF-matched | q6L4 functional (88) | 97 | 0.1 | 0.3425 +- 0.1174 | 0.7213 +- 0.2003 | 0.3968 |
| step | MLP-matched | q6L4 functional (88) | 93 | 0.03 | 0.1710 +- 0.0267 | 0.1993 +- 0.0391 | 0.1644 |

## Ablation: near-surface exclusion width x_min of the initial-rate condition (exp pulse, gamma 9, seed 0, E1 recipe, N_c 512, E1-selected learning rates; the x_min = 0.3 rows of the circuit and the matched MLP are their E1 seed-0 runs)
| model | params | lr | x_min | T relL2 | sigma relL2 |
|---|---|---|---|---|---|
| MLP-ref | 12802 | 0.01 | 0 | 0.0176 | 0.1001 |
| MLP-ref | 12802 | 0.01 | 0.1 | 0.0143 | 0.0705 |
| MLP-ref | 12802 | 0.01 | 0.3 | 0.0085 | 0.0496 |
| MLP-ref | 12802 | 0.01 | 0.6 | 0.0166 | 0.0617 |
| MLP-ref | 12802 | 0.01 | 1 | 0.0082 | 0.0732 |
| QPINN (VQC) | 124 | 0.1 | 0 | 0.1824 | 0.9644 |
| QPINN (VQC) | 124 | 0.1 | 0.1 | 0.1709 | 1.1081 |
| QPINN (VQC) | 124 | 0.1 | 0.3 | 0.1939 | 0.8785 |
| MLP-matched | 114 | 0.1 | 0 | 0.1330 | 0.2695 |
| MLP-matched | 114 | 0.1 | 0.1 | 0.1294 | 0.2393 |
| MLP-matched | 114 | 0.1 | 0.3 | 0.1959 | 0.3409 |

## Secondary wave speeds (exp pulse, three seeds; VQC 6q4L, nominal-matched MLP and FF, MLP-ref; E1 recipe with the gamma-9 exp-pulse rates: 0.1, MLP-ref 0.01)
| gamma | model | params | lr | T relL2 mean +- sd | sigma relL2 mean +- sd | T per seed | sigma per seed | train s mean |
|---|---|---|---|---|---|---|---|---|
| 25 | FF-matched | 116 | 0.1 | 0.3180 +- 0.0970 | 0.6342 +- 0.3302 | 0.3349, 0.2136, 0.4054 | 0.5232, 1.0056, 0.3738 | 56 |
| 25 | MLP-matched | 114 | 0.1 | 0.1279 +- 0.0525 | 0.2353 +- 0.1525 | 0.1595, 0.0674, 0.1569 | 0.2019, 0.1023, 0.4017 | 21 |
| 25 | MLP-ref | 12802 | 0.01 | 0.0221 +- 0.0028 | 0.1044 +- 0.0357 | 0.0221, 0.0193, 0.0250 | 0.1237, 0.0632, 0.1264 | 239 |
| 25 | QPINN (VQC) | 124 | 0.1 | 0.2927 +- 0.0458 | 1.1971 +- 0.4658 | 0.3175, 0.2399, 0.3208 | 1.7249, 0.8438, 1.0225 | 4116 |
| 49 | FF-matched | 116 | 0.1 | 0.4400 +- 0.1898 | 0.6894 +- 0.2043 | 0.6394, 0.2615, 0.4190 | 0.9249, 0.5832, 0.5600 | 47 |
| 49 | MLP-matched | 114 | 0.1 | 0.2468 +- 0.0621 | 0.4511 +- 0.1311 | 0.1962, 0.2281, 0.3161 | 0.3184, 0.5806, 0.4543 | 19 |
| 49 | MLP-ref | 12802 | 0.01 | 0.0421 +- 0.0014 | 0.0956 +- 0.0208 | 0.0436, 0.0417, 0.0408 | 0.1029, 0.1117, 0.0721 | 214 |
| 49 | QPINN (VQC) | 124 | 0.1 | 0.3868 +- 0.1884 | 1.4485 +- 0.8240 | 0.3257, 0.5982, 0.2366 | 2.3413, 1.2869, 0.7172 | 5454 |

## E2: gradient variance at random initialisation (exp pulse, gamma = 9, 200 initialisations, first-layer RY angle on qubit 0)

| L | loss | n_q=2 | n_q=3 | n_q=4 | n_q=5 | n_q=6 | n_q=7 | n_q=8 | n_q=9 | n_q=10 | fitted slope d log10 Var / d n_q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | data | 9.56e-01 | 7.54e-01 | 5.05e-01 | 7.07e-01 | 6.00e-01 | 7.83e-01 | 6.61e-01 | 6.21e-01 | 7.22e-01 | -0.008 |
| 2 | physics | 5.89e+00 | 9.31e+00 | 4.66e+00 | 7.47e+00 | 1.05e+01 | 8.65e+00 | 3.43e+00 | 1.10e+01 | 1.47e+01 | 0.027 |
| 2 | combined | 9.01e+00 | 1.25e+01 | 6.54e+00 | 1.03e+01 | 1.36e+01 | 1.14e+01 | 5.88e+00 | 1.37e+01 | 1.72e+01 | 0.020 |
| 4 | data | 2.94e-01 | 1.00e-01 | 6.45e-02 | 6.59e-02 | 7.77e-02 | 5.62e-02 | 7.28e-02 | 7.19e-02 | 9.93e-02 | -0.038 |
| 4 | physics | 4.87e+01 | 2.79e+01 | 9.30e+00 | 2.48e+01 | 7.30e+00 | 1.65e+01 | 1.08e+01 | 1.17e+01 | 7.04e+00 | -0.076 |
| 4 | combined | 5.05e+01 | 2.89e+01 | 9.68e+00 | 2.53e+01 | 7.80e+00 | 1.70e+01 | 1.15e+01 | 1.23e+01 | 7.62e+00 | -0.074 |
| 8 | data | 1.41e-01 | 3.26e-02 | 1.50e-02 | 5.67e-03 | 3.84e-03 | 2.48e-03 | 2.20e-03 | 2.30e-03 | 2.73e-03 | -0.206 |
| 8 | physics | 3.83e+02 | 2.76e+02 | 1.25e+02 | 5.28e+01 | 2.13e+01 | 8.72e+00 | 1.02e+01 | 8.99e+00 | 1.06e+01 | -0.227 |
| 8 | combined | 3.85e+02 | 2.78e+02 | 1.26e+02 | 5.30e+01 | 2.14e+01 | 8.80e+00 | 1.03e+01 | 9.09e+00 | 1.07e+01 | -0.227 |

## E4: inference under finite shots and depolarising noise (best two seeds per case, gamma and configuration; 41 x 40 subgrid; per-layer single-qubit depolarising p, binomial shot sampling of each <Z_k>; rows with p = 0 and shots = exact are the clean same-grid controls)
| case | gamma | config | p | shots | T relL2 mean +- sd | sigma relL2 mean +- sd | n |
|---|---|---|---|---|---|---|---|
| exp | 9 | q4L2 | 0 | exact | 0.3308 +- 0.0968 | 0.9517 +- 0.1331 | 2 |
| exp | 9 | q4L2 | 0 | 1000 | 73.8240 +- 67.6805 | 21.3149 +- 20.9250 | 2 |
| exp | 9 | q4L2 | 0 | 10000 | 22.4574 +- 20.3642 | 6.9947 +- 6.7734 | 2 |
| exp | 9 | q4L2 | 0 | 100000 | 7.1757 +- 6.3994 | 2.4748 +- 1.9383 | 2 |
| exp | 9 | q4L2 | 0.001 | exact | 0.3290 +- 0.0966 | 0.9500 +- 0.1333 | 2 |
| exp | 9 | q4L2 | 0.001 | 1000 | 73.9079 +- 68.5329 | 21.3403 +- 20.8135 | 2 |
| exp | 9 | q4L2 | 0.001 | 10000 | 22.3959 +- 20.3455 | 7.0580 +- 6.8084 | 2 |
| exp | 9 | q4L2 | 0.001 | 100000 | 7.2117 +- 6.4485 | 2.4538 +- 1.9118 | 2 |
| exp | 9 | q4L2 | 0.01 | exact | 0.3133 +- 0.0949 | 0.9346 +- 0.1350 | 2 |
| exp | 9 | q4L2 | 0.01 | 1000 | 73.8193 +- 68.6636 | 21.3438 +- 20.8092 | 2 |
| exp | 9 | q4L2 | 0.01 | 10000 | 22.5016 +- 20.4681 | 6.9110 +- 6.7321 | 2 |
| exp | 9 | q4L2 | 0.01 | 100000 | 7.1371 +- 6.4137 | 2.4750 +- 1.9450 | 2 |
| exp | 9 | q6L4 | 0 | exact | 0.1595 +- 0.0309 | 0.5804 +- 0.0166 | 2 |
| exp | 9 | q6L4 | 0 | 1000 | 2.2119 +- 0.0068 | 2.5758 +- 0.2728 | 2 |
| exp | 9 | q6L4 | 0 | 10000 | 0.7053 +- 0.0582 | 0.9735 +- 0.0950 | 2 |
| exp | 9 | q6L4 | 0 | 100000 | 0.2691 +- 0.0321 | 0.6299 +- 0.0251 | 2 |
| exp | 9 | q6L4 | 0.001 | exact | 0.1556 +- 0.0282 | 0.5753 +- 0.0184 | 2 |
| exp | 9 | q6L4 | 0.001 | 1000 | 2.1775 +- 0.1358 | 2.6298 +- 0.3841 | 2 |
| exp | 9 | q6L4 | 0.001 | 10000 | 0.7064 +- 0.0501 | 0.9714 +- 0.0901 | 2 |
| exp | 9 | q6L4 | 0.001 | 100000 | 0.2721 +- 0.0251 | 0.6185 +- 0.0168 | 2 |
| exp | 9 | q6L4 | 0.01 | exact | 0.1588 +- 0.0336 | 0.6135 +- 0.1210 | 2 |
| exp | 9 | q6L4 | 0.01 | 1000 | 2.2078 +- 0.0967 | 2.5795 +- 0.3999 | 2 |
| exp | 9 | q6L4 | 0.01 | 10000 | 0.7039 +- 0.0430 | 0.9982 +- 0.1689 | 2 |
| exp | 9 | q6L4 | 0.01 | 100000 | 0.2697 +- 0.0008 | 0.6630 +- 0.1217 | 2 |
| smooth | 9 | q4L2 | 0 | exact | 0.3249 +- 0.0189 | 0.9044 +- 0.0154 | 2 |
| smooth | 9 | q4L2 | 0 | 1000 | 73.0238 +- 101.5526 | 8.6281 +- 10.8824 | 2 |
| smooth | 9 | q4L2 | 0 | 10000 | 23.3152 +- 32.2638 | 2.9909 +- 2.9338 | 2 |
| smooth | 9 | q4L2 | 0 | 100000 | 7.5194 +- 10.1558 | 1.3653 +- 0.6361 | 2 |
| smooth | 9 | q4L2 | 0.001 | exact | 0.3898 +- 0.1111 | 0.9046 +- 0.0157 | 2 |
| smooth | 9 | q4L2 | 0.001 | 1000 | 73.0445 +- 101.5041 | 8.6282 +- 10.8859 | 2 |
| smooth | 9 | q4L2 | 0.001 | 10000 | 23.9366 +- 33.1582 | 3.0050 +- 2.9535 | 2 |
| smooth | 9 | q4L2 | 0.001 | 100000 | 7.3960 +- 9.9820 | 1.3765 +- 0.6522 | 2 |
| smooth | 9 | q4L2 | 0.01 | exact | 1.2674 +- 1.3552 | 0.9069 +- 0.0185 | 2 |
| smooth | 9 | q4L2 | 0.01 | 1000 | 74.1424 +- 103.1490 | 8.4736 +- 10.6587 | 2 |
| smooth | 9 | q4L2 | 0.01 | 10000 | 23.3594 +- 32.3435 | 3.0202 +- 2.9673 | 2 |
| smooth | 9 | q4L2 | 0.01 | 100000 | 7.4067 +- 9.9998 | 1.4025 +- 0.6822 | 2 |
| smooth | 9 | q6L4 | 0 | exact | 0.1467 +- 0.0334 | 0.5027 +- 0.0420 | 2 |
| smooth | 9 | q6L4 | 0 | 1000 | 1.2773 +- 0.7026 | 1.3724 +- 0.0278 | 2 |
| smooth | 9 | q6L4 | 0 | 10000 | 0.4320 +- 0.2159 | 0.6369 +- 0.0212 | 2 |
| smooth | 9 | q6L4 | 0 | 100000 | 0.1949 +- 0.0692 | 0.5163 +- 0.0425 | 2 |
| smooth | 9 | q6L4 | 0.001 | exact | 0.1539 +- 0.0310 | 0.5041 +- 0.0386 | 2 |
| smooth | 9 | q6L4 | 0.001 | 1000 | 1.2904 +- 0.7148 | 1.3423 +- 0.0080 | 2 |
| smooth | 9 | q6L4 | 0.001 | 10000 | 0.4415 +- 0.2206 | 0.6299 +- 0.0275 | 2 |
| smooth | 9 | q6L4 | 0.001 | 100000 | 0.2015 +- 0.0657 | 0.5175 +- 0.0384 | 2 |
| smooth | 9 | q6L4 | 0.01 | exact | 0.2357 +- 0.0060 | 0.5319 +- 0.0026 | 2 |
| smooth | 9 | q6L4 | 0.01 | 1000 | 1.3081 +- 0.7323 | 1.3870 +- 0.0219 | 2 |
| smooth | 9 | q6L4 | 0.01 | 10000 | 0.4791 +- 0.1837 | 0.6600 +- 0.0052 | 2 |
| smooth | 9 | q6L4 | 0.01 | 100000 | 0.2708 +- 0.0312 | 0.5473 +- 0.0029 | 2 |
| step | 9 | q4L2 | 0 | exact | 0.6153 +- 0.1755 | 0.8754 +- 0.0222 | 2 |
| step | 9 | q4L2 | 0 | 1000 | 14.8745 +- 17.8301 | 7.5606 +- 7.1462 | 2 |
| step | 9 | q4L2 | 0 | 10000 | 4.7963 +- 5.5767 | 2.5643 +- 2.0159 | 2 |
| step | 9 | q4L2 | 0 | 100000 | 1.6668 +- 1.5980 | 1.2036 +- 0.3992 | 2 |
| step | 9 | q4L2 | 0.001 | exact | 0.6134 +- 0.1769 | 0.8710 +- 0.0171 | 2 |
| step | 9 | q4L2 | 0.001 | 1000 | 14.8401 +- 17.8334 | 7.5581 +- 7.1904 | 2 |
| step | 9 | q4L2 | 0.001 | 10000 | 4.7419 +- 5.4846 | 2.5953 +- 2.0907 | 2 |
| step | 9 | q4L2 | 0.001 | 100000 | 1.6123 +- 1.5334 | 1.2206 +- 0.4446 | 2 |
| step | 9 | q4L2 | 0.01 | exact | 0.5974 +- 0.1906 | 0.8566 +- 0.0056 | 2 |
| step | 9 | q4L2 | 0.01 | 1000 | 14.7380 +- 17.7327 | 7.5403 +- 7.1826 | 2 |
| step | 9 | q4L2 | 0.01 | 10000 | 4.7569 +- 5.5249 | 2.5909 +- 2.0626 | 2 |
| step | 9 | q4L2 | 0.01 | 100000 | 1.6660 +- 1.6301 | 1.1910 +- 0.4320 | 2 |
| step | 9 | q6L4 | 0 | exact | 0.2261 +- 0.0102 | 0.7839 +- 0.0576 | 2 |
| step | 9 | q6L4 | 0 | 1000 | 1.0864 +- 0.1215 | 1.2875 +- 0.3769 | 2 |
| step | 9 | q6L4 | 0 | 10000 | 0.3964 +- 0.0344 | 0.8455 +- 0.0039 | 2 |
| step | 9 | q6L4 | 0 | 100000 | 0.2514 +- 0.0043 | 0.7912 +- 0.0535 | 2 |
| step | 9 | q6L4 | 0.001 | exact | 0.2214 +- 0.0053 | 0.7749 +- 0.0519 | 2 |
| step | 9 | q6L4 | 0.001 | 1000 | 1.0987 +- 0.1280 | 1.2799 +- 0.3607 | 2 |
| step | 9 | q6L4 | 0.001 | 10000 | 0.3963 +- 0.0399 | 0.8458 +- 0.0211 | 2 |
| step | 9 | q6L4 | 0.001 | 100000 | 0.2488 +- 0.0017 | 0.7817 +- 0.0465 | 2 |
| step | 9 | q6L4 | 0.01 | exact | 0.2029 +- 0.0140 | 0.7339 +- 0.0261 | 2 |
| step | 9 | q6L4 | 0.01 | 1000 | 1.0939 +- 0.2022 | 1.2507 +- 0.3792 | 2 |
| step | 9 | q6L4 | 0.01 | 10000 | 0.3936 +- 0.0455 | 0.8039 +- 0.0761 | 2 |
| step | 9 | q6L4 | 0.01 | 100000 | 0.2300 +- 0.0181 | 0.7407 +- 0.0318 | 2 |


# Functional (output-affecting) parameter counts of the circuits (numerical perturbation test, 21 Oct 2026)
Each parameter was perturbed by an additive +0.37 in its native units (radians for rotation angles) at random initialisation and the two outputs re-evaluated on 64 random inputs;
a parameter is inert if the outputs did not change (tolerance 1e-12). With Z readout on qubits 0 and 1 after a diagonal CZ
ring, every single-qubit gate of the last layer on the unmeasured qubits, the last-layer final RZ on the measured qubits,
and some RZ angles of the penultimate layer on unmeasured qubits cannot change the outputs.

| circuit | nominal | inert (encoding, rotation) | functional |
|---|---|---|---|
| 4q2L | 44 | 14 (4, 10) | 30 |
| 6q4L | 124 | 36 (12, 24) | 88 |
| 4q4L | 84 | 14 (4, 10) | 70 |
| 6q2L | 64 | 34 (12, 22) | 30 |
| 6q8L | 244 | 36 (12, 24) | 208 |
| 8q2L | 84 | 54 (20, 34) | 30 |

The E1 matched nets exceed the circuits' functional counts by +40.0 percent (MLP 42 vs 30), +33.3 percent (FF 40 vs 30),
+29.5 percent (MLP 114 vs 88) and +31.8 percent (FF 116 vs 88). A second set of classical nets matched to the functional counts
(30 and 88) was trained with the same protocol (run_functional_matched.sh; results below and table E1b).

## Functional-matched classical nets (results/fmatch; table "E1b" in the package)
Nearest achievable counts: MLP 29 (target 30, -3 percent) and 93 (target 88, +6 percent); FF-matched 97 (target 88, +10 percent).
For target 30 the smallest Fourier-feature width searched was h = 2 (40 parameters; the formula 19h + 2 would give 21 at h = 1,
which was not searched), so its "functional-matched" 4q2L row is the same 40-parameter network as in E1 (identical numbers) and is not a valid
functional match; the MLP rows are the functional-matched comparison. Learning rates were re-selected per cell by validation.
Outcome: the MLP matched to the functional count still has lower stress error than the circuit in all six cells
(exp 0.44 vs 1.11 and 0.29 vs 0.76; step 0.77 vs 1.19 and 0.20 vs 0.90; smooth 0.49 vs 0.98 and 0.36 vs 0.57;
mean of five seeds, circuit values from E1).


# Ablation of the initial-rate condition (single seed 0, exp pulse, gamma = 9, mixed sampling, N_c = 1024, lr 3e-3, Adam 2000 + L-BFGS 1000; same code as E1 except these settings; logs/variants/V1-V6)

| model | params | initial-rate weight | T relL2 | sigma relL2 |
|---|---|---|---|---|
| MLP-ref | 12802 | 0 | 0.0155 | 0.2152 |
| MLP-ref | 12802 | 1 | 0.0124 | 0.0492 |
| MLP-ref | 12802 | 0.1 | 0.0169 | 0.0599 |
| MLP-matched | 114 | 0 | 0.1322 | 0.8132 |
| MLP-matched | 114 | 1 | 0.1867 | 0.1827 |
| MLP-matched | 114 | 0.1 | 0.0846 | 0.2688 |

Weight 0 removes the condition (the stress problem is then under-determined); weight 1 is the E1 setting.


# Output gains of the E4 selected checkpoints (alpha_T, alpha_sigma, beta_T, beta_sigma from the saved state dicts)

| checkpoint | case | config | seed | clean sigma relL2 (E1 grid) | alpha_T | alpha_sigma | beta_T | beta_sigma |
|---|---|---|---|---|---|---|---|---|
| vqc_exp_g9_physics_q4L2_s0 | exp | q4L2 | 0 | 0.8460 | 43.90 | 37.31 | 0.015 | 0.046 |
| vqc_exp_g9_physics_q4L2_s3 | exp | q4L2 | 3 | 1.0487 | 198.51 | 207.61 | 0.000 | 0.057 |
| vqc_exp_g9_physics_q6L4_s1 | exp | q6L4 | 1 | 0.5919 | 3.55 | 15.60 | 0.003 | -0.207 |
| vqc_exp_g9_physics_q6L4_s3 | exp | q6L4 | 3 | 0.5698 | 3.80 | 12.67 | 0.027 | -0.254 |
| vqc_smooth_g9_physics_q4L2_s2 | smooth | q4L2 | 2 | 0.8986 | -236.17 | 82.77 | -4.051 | 0.025 |
| vqc_smooth_g9_physics_q4L2_s4 | smooth | q4L2 | 4 | 0.9207 | -1.96 | 0.95 | 0.050 | -0.135 |
| vqc_smooth_g9_physics_q6L4_s1 | smooth | q6L4 | 1 | 0.5300 | 2.95 | 6.45 | -0.007 | 0.138 |
| vqc_smooth_g9_physics_q6L4_s4 | smooth | q6L4 | 4 | 0.4760 | 1.29 | 6.53 | 0.003 | -0.261 |
| vqc_step_g9_physics_q4L2_s1 | step | q4L2 | 1 | 0.8599 | 50.73 | 113.36 | 0.138 | -0.184 |
| vqc_step_g9_physics_q4L2_s2 | step | q4L2 | 2 | 0.8922 | 3.96 | 20.59 | 0.099 | -3.246 |
| vqc_step_g9_physics_q6L4_s0 | step | q6L4 | 0 | 0.8237 | 1.78 | 5.76 | -0.027 | -0.444 |
| vqc_step_g9_physics_q6L4_s1 | step | q6L4 | 1 | 0.7391 | 2.20 | 11.89 | 0.037 | 0.854 |


# Figure captions (results_package/figures)

fig_e1_errors.pdf — E1 parity test at matched parameter count. Relative L2 error of temperature (left) and stress (right)
on the 241 x 240 test grid for the three pulses (exp, step, smooth), gamma = 9, physics-only training, identical budget
(Adam 2000 + L-BFGS 1000, 512 collocation points per step). Bars: mean over five seeds, error bars: sample standard
deviation. Red: QPINN (variational circuit), blue: MLP-matched, green: Fourier-feature MLP-matched; solid bars are the
6-qubit, 4-layer configuration (124 parameters) and its matched nets (114, 116), hatched bars the 4-qubit, 2-layer
configuration (44) and its matched nets (42, 40); black: 12,802-parameter reference MLP.

fig_fields_exp.pdf — Exact and learned fields for the exponential pulse, gamma = 9. Top row: temperature T(x,t);
bottom row: stress sigma(x,t); columns: exact solution, best-seed QPINN (6 qubits, 4 layers), best-seed matched MLP,
reference MLP. The stress wave front x = 3t separates the compressive band from the relaxed region; the physics-trained circuit
misses the band at 124 parameters (its supervised E3 runs with the same architecture reach a stress error of 0.22, so this is
a training outcome, not a representational bound).

fig_e2_gradvar.pdf — E2 gradient variance at random initialisation versus qubit count for circuit depths L = 2, 4, 8
(200 initialisations; derivative of the loss with respect to the first-layer RY angle on qubit 0; exponential pulse).
Blue: data loss (MSE on 256 exact samples), red: physics loss, purple: combined. The physics loss raises the variance by factors
5.2-20.3 (L = 2), 70.9-375.4 (L = 4) and 2726-9312 (L = 8); fitted slopes of log10 Var per qubit over n_q = 2-10: data
-0.008/-0.038/-0.206 and physics +0.027/-0.076/-0.227 for L = 2/4/8, with the L = 8 tail (n_q = 7-10) at +0.014 (data) and
+0.020 (physics); no slope-equivalence test was performed.

fig_e3_pareto.pdf — E3 stress error versus trainable parameter count (exp pulse, gamma = 9, mean of seeds 0 and 1) for the
circuits (4,2), (4,4), (6,2), (6,4), (6,8), (8,2) and their nominal-count-matched MLP and Fourier-feature nets. Red filled
circles: circuits at their nominal count; red hollow squares: the same circuits plotted at their functional (output-affecting)
count; blue: MLP; green: Fourier-feature MLP.

fig_e3_colloc.pdf — E3 stress error versus collocation points per Adam step N_c (physics-only; L-BFGS on 4 N_c fixed points)
for the 6-qubit, 4-layer circuit (N_c = 64, 256, 512, 1024) and its nominal-count-matched MLP (also N_c = 4096); mean of two
seeds. The circuit at N_c = 4096 was dropped for cost (about nine hours per seed).

fig_e3_data.pdf — E3 stress error versus the number of labelled exact samples N_d (data-only training: MSE on N_d fixed
samples, no PDE loss, Adam 2000 steps, no L-BFGS; learning rates 0.03 for the circuit and 0.1 for the MLP, selected by
validation in this regime) for the same two models; mean of two seeds.


# Depth-2 output-family equivalence check (family_equivalence.py, executed 21 Oct 2026)

family equivalence test: 64 random inputs (x in [0,6], t in [0,3]); float64/complex128 statevector
L=2: 4-qubit parameters transplanted onto qubits (0,1,2,5) of the 6-qubit circuit: max |output difference| = 5.55e-16 (identical function family)
L=2: 4-qubit parameters transplanted onto qubits (0,1,2,7) of the 8-qubit circuit: max |output difference| = 8.88e-16 (identical function family)
L=4: 4-qubit parameters transplanted onto qubits (0,1,2,5) of the 6-qubit circuit: max |output difference| = 8.11e-01 (different: control)
L=4: 4-qubit parameters transplanted onto qubits (0,1,2,7) of the 8-qubit circuit: max |output difference| = 8.48e-01 (different: control)
