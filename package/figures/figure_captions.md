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
