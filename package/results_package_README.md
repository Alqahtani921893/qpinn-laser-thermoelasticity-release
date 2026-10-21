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
