# Testing the value of smoothing

This second experiment campaign compares all methods at a **common original
primal-dual gap**. It retains the earlier archive and its negative results.
The new methods are deterministic implementations, not the authors' software
from the neighboring papers.

## Run

From the repository root, with the pinned requirements installed:

```bash
python experiments/value_benchmark.py check --output runs/value_checks
python experiments/value_benchmark.py pilot --output runs/value_pilot
python experiments/value_benchmark.py suite --output runs/value_review
python experiments/analyze_value_benchmark.py --output runs/value_review
```

The first command checks block derivatives, smoothed model solutions, the
Moreau elimination identity, and five solvers on both regularizers. The pilot
runs two configurations before the full campaign. The suite resumes existing
per-run files; use a new directory for a clean rerun. Never reuse partial
results produced by a different code version. The analysis recomputes every
available final gap, reports failures, and generates the three final figure
groups and full tables. Its terminal plotting point uses the measured total
time and final gap, including unsuccessful inner work.

The suite's preliminary two plots are replaced by the final analysis plots.
The final three figures are `value_curves`, `structure_curves`, and `scaling`
in both PDF and PNG format. No curve is fitted or averaged.

## Design

The 11 configurations vary dimension (80, 400, 1200, 2400), feature correlation,
ridge strength, regularizer strength, target accuracy, and regularizer structure.
All have 600 observations and seed zero. The second structure is a sum of
nonoverlapping size-five Euclidean norms. It has an explicit block proximal
map and radial/tangential base solves. The conditioning stress changes both
correlation and ridge strength; it does not isolate those factors.

The main methods are MEL, continuation MEL (CONT), direct nonsmooth PQN,
direct nonsmooth PN, R-FISTA, and smoothed exact-Hessian MEL-PN. Dense PN and
MEL-PN are not tested above n=400, as a declared resource limit. This is not
evidence that all Newton implementations are expensive: matrix-free and
data-rank Newton methods are not evaluated.

PQN uses the same BFGS curvature on the original nonsmooth model. Its inner
proximal-residual SSN uses a proximal-gradient descent safeguard and a true
subgradient-residual check at the returned point. The model tolerance cap is
the same as for the smoothed solvers. This baseline is a representative
implementation of the proximal Newton-type framework, not a claim of the
best available implementation of that framework.

CONT starts at max(0.01, epsilon/Lh²), divides alpha by ten when its stage
gap is at most alpha Lh²/4, and retains curvature pairs across stages. Every
stage is included in time and counters. It can stop as soon as the original
certificate passes. MEL starts directly with alpha = epsilon/Lh².

At n=400, scalar metric, gradient-inner, and memory-5/20 ablations isolate
curvature, inner solver, and memory choices. The gradient-inner method uses
the current metric upper bound, not a deliberately inflated bound. An inner
failure is not a speedup or a completed solution.

## Measurement and limits

- One same-setting warm-up per fresh worker, then one measured run; three
  workers per setting. Method order rotates. Warm-up may terminate at its
  budget; it does not supply the measured starting point.
- The clock starts after the common zero-vector/counter initialization.
  Metric preparation, spectral checks, certificates, gradients, backtracking,
  inner work, all stages, and trace collection are included. Data preparation,
  exact spectral norm computation, imports, process startup and file output
  are outside the solve timer for every method.
- One BLAS thread, double precision, identical data and zero initial points.
  Report median and minimum--maximum; three timing repeats are not independent
  data seeds or a significance test.
- Measured soft budget: 45 s checked between outer iterations. Warm-up: 20 s.
  Inner cap: 1000; outer model cap: 600; R-FISTA cap: 15000.
  A worker hard timeout of 150 s includes preparation and warm-up.
- Peak memory is OS peak resident process memory, including imports, data
  preparation and warm-up. It is not isolated metric storage or a sampled
  Python-allocation estimate.
- Compact runs never silently recover through a dense n-by-n solve.
  Arithmetic and budget failures are retained. Strict model tolerances can
  defeat otherwise useful solvers; these outcomes apply to this implementation.
- All final gaps are double-precision primal-dual certificates, not validated
  interval bounds. The exact-arithmetic local rate conditions remain separate.

Raw records include matrix products, gradients, Hessian constructions,
certificates, spectral checks, metric actions, reduced/dense linear solves,
backtracks, curvature skips, metric resets, SSN or gradient inner steps,
the maximum SSN linear and directional residual ratios, and failure status.

No stochastic benchmark was run. A follow-up must account for sample gradients,
full-gradient refreshes, independent sampling seeds, separate tuning instances,
and the cost/timing of common full-data certificates. No performance comparison
with Song et al.'s stochastic method is implied by this deterministic campaign.
