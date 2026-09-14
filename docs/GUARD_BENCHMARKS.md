# Safeguard and exact-envelope follow-up

This release adds two separately measured batches. Do not pool them or earlier versions. Every formal setting has three independent workers, a two-second warm-up, a 45-second measured budget, and common original primal-dual tolerance 1e-6. Method order rotates between repetitions. Data seeds and parameters match the four declared cases: s4000, s16000, weak4000 and group8000. Setup and the shared spectral bound are outside every solve timer. All continuation stages, certificates, metric checks and discarded trials are inside it. No dense fallback is used.

## Methods

- P-CONT is the unchanged archived reference-value safeguard.
- P-CONT-C caches the trial objective, avoiding a second evaluation of the accepted trial.
- P-CONT-D also uses the descent-lemma threshold and reuses the current value. Both C and D retain failure replacement and finite trial budgets. The added corollary covers D; it does not claim a new asymptotic rate.
- PQN, R-FISTA and PN-MF are unchanged archived implementations.
- FBE-LBFGS is our implementation of Stella, Themelis and Patrinos, COAP 67 (2017), Algorithm 2. gamma=.95/L; memory ten; curvature pairs use the envelope gradients at x and the accepted line-search point w; the next iterate is the FB point at w. Gradient evaluation uses exact logistic Hessian-vector products. Failed envelope search falls back to the FB step at x. This is not author-software reproduction. Original-gap stopping, 1200 updates and 45-second budgets apply. No nested subproblem or smoothing continuation is used.

## Run from the repository root

Use Python 3.12 and install requirements.txt. Single-thread settings are set by the scripts. Fresh output directories are required to avoid mixing protocols or replacing records.

```bash
python experiments/guard_benchmark.py pilot --output runs/guard_pilot --data-dir runs/guard_data
python experiments/analyze_guard_benchmark.py --output runs/guard_pilot --data-dir runs/guard_data
python experiments/guard_benchmark.py suite --output runs/guard_formal --data-dir runs/guard_data
python experiments/analyze_guard_benchmark.py --output runs/guard_formal --data-dir runs/guard_data
python experiments/fbe_benchmark.py check --output runs/fbe_checks --data-dir runs/guard_data
python experiments/fbe_benchmark.py pilot --output runs/fbe_pilot --data-dir runs/guard_data
python experiments/analyze_guard_benchmark.py --output runs/fbe_pilot --data-dir runs/guard_data
python experiments/fbe_benchmark.py suite --output runs/fbe_formal --data-dir runs/guard_data
python experiments/analyze_guard_benchmark.py --output runs/fbe_formal --data-dir runs/guard_data
python reproduce.py verify
```

Run suites sequentially, with one benchmark worker at a time. Data arrays regenerate from the case parameters. Protocols store cached-array and source hashes. Binary NPZ hashes need not match across fresh ZIP packaging or library versions; the source, parameters and data construction are recorded. Existing protocol/data hashes must match within an audit. No network dataset is required by these follow-up cases.

## Audit and interpretation

Raw terminal JSON stores the returned vector, history, full time, peak process memory, operation counts, status and failures. The independent audit recomputes the original dual gap with a separate explicit formula. It also checks recorded acceptance margins for C and D, but intermediate vectors are not retained; these margin checks are not independent reevaluations of the whole trajectory or interval-arithmetic certificates.

Curve panels show actual median-total-time runs, with all terminal time and failed work included. They are neither fitted nor averaged trajectories. The peak memory is whole-process working set, including data, imports and warm-up, not isolated metric storage. Full minimum–maximum ranges and counts are reported. A shared desktop and time-based trial caps limit timing precision. The guard batch overlapped with manuscript editing and one LaTeX compilation; no dedicated-hardware or statistical-significance claim is made. Failed times must not be treated as solution times.

Published v1, v2.0 and v2.1 source files and results remain unchanged. CI verifies archive hashes and stored-result consistency; it does not rerun numerical experiments. The research audit explicitly separates a proved implementation improvement from unresolved publication-level originality.
