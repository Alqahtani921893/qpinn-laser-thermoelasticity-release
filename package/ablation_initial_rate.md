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
