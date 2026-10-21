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
