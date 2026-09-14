# Expanded scale and projected continuation campaign

The v2 archives are immutable. This campaign uses `experiments/scale_benchmark.py` and a separate result directory. It adds 4,000/8,000/16,000 variables, a second data seed, isolated correlation changes at fixed ridge strength, weak regularization, tighter tolerance, nonoverlapping groups, and 2,000 samples.

```bash
python experiments/scale_benchmark.py check --output runs/scale_checks
python experiments/scale_benchmark.py pilot --output runs/scale_pilot
python experiments/check_projected_theory.py --output runs/projected_theory
python experiments/scale_benchmark.py suite --output runs/scale_review
python experiments/analyze_scale_benchmark.py --output runs/scale_review
```

The suite generates and caches deterministic synthetic arrays under `data/`. This preparation, including the common exact spectral norm, is excluded from all solve timers. Array caches can be regenerated from the recorded seeds and parameters. The protocol contains cache and timed-source hashes. Use an empty directory after any code change; completed per-run files are reused on resume.

## Methods

- CONT: the earlier smoothed-gap continuation schedule, returning the unprojected iterate.
- CONT-P: the same schedule and model, but check and return the proximal point. This isolates primal recovery from the new safeguard and stage rule.
- P-CONT: return the proximal point, reduce alpha when the smoothed gradient norm is at most alpha, and compare every model trial against a gradient reference step with step size 1/(Lf+1/alpha). A failed or worse trial is replaced by the reference. Compare each warm start with the zero anchor under the new alpha. It records the actual projected residual certificate and all added evaluations. The algorithm can stop earlier on the same valid original primal-dual gap used by every method.
- PQN: the direct nonsmooth compact model, with the same curvature and strict model residual criterion.
- PN-MF: exact loss Hessian through matrix-vector products, with restarted accelerated proximal-gradient inner solves. The stopping residual is a true model subgradient at the returned point. No n-by-n Hessian is allocated. This is a representative implementation, not an optimized external package or the fastest possible PN solver.
- R-FISTA: direct nonsmooth restarted accelerated proximal gradient.
- MEL: direct small smoothing parameter, included on the 4,000-variable reference case.

Only P-CONT has the reference decrease guarantee and the new work theorem. Neither the squared projected-stage error nor this safeguard establishes fixed-memory outer superlinear convergence. An alpha-independent worst-case trial budget is required in the theorem; the implementation has 1,000 inner iterations, 50 inner/outer backtracks, and an additional earlier time cutoff. Failure of a P-CONT model trial is counted even when the whole solve later succeeds.

## Fairness and accounting

All methods use the same zero initial point, data, original primal-dual gap and single BLAS thread. There is one sequential measured worker at a time. Each setting has three fresh workers, each with a two-second same-setting warm-up and a 45-second measured soft budget. Method order rotates. Model methods have 1,200 outer and 1,000 inner iteration caps; R-FISTA has 15,000 updates. Inner work checks the deadline, rather than silently overrunning it. Each subprocess has a 120-second hard timeout, including loading and warm-up. The P-CONT trial timeout is at most two seconds; it does not stop the reference mechanism.

Report measured total solve time, not time after the last smoothing stage. Counters include all stage checks, original certificates, reference objectives, model trials, Hessian-vector products, guards, resets and failures. The peak is the operating-system process working-set high-water mark, including imports, cached data loading and warm-up; it is not isolated solver memory. Three repeats measure timing variation, not three independent data seeds. A second seed is an explicit separate case. All gaps are double-precision checks, not interval certificates.

The later campaign uses a different warm-up and outer budget from v2; do not pool its times with v2 times. Failed times are consumed budgets and must not enter successful time-to-solution rankings. No stochastic baseline is included, and no claim against Song et al.'s stochastic method follows.

## Follow-up safeguard ablation

```bash
python experiments/scale_ablation.py suite --output runs/scale_ablation --data-dir runs/scale_review/data
python experiments/analyze_scale_benchmark.py --output runs/scale_review --ablation runs/scale_ablation
```

P-CONT-NG retains projected recovery and the residual-stage rule, but removes the gradient reference and its failure replacement. It uses the full common inner/overall budgets. This separates the stage-rule change (CONT-P versus P-CONT-NG) from the reference safeguard (P-CONT-NG versus P-CONT). It is run on s4000, s16000, weak4000 and group8000, with three measured repetitions each, after the main batch. It has no safeguarded-work guarantee. Small timing differences across these consecutive batches should not be read as significant; operation counts and large effects are more informative.

## Interpretation limits

The dimension series regenerates matrices and labels at each size using the stated seed; it is not a nested common-label design and cannot identify a pure arithmetic complexity exponent. Correlation comparisons keep ridge strength and the relative regularization coefficient fixed, while the calibrated absolute lambda may change. Data preparation and the common spectral bound are outside every solve timer, so the reported seconds are not from-file-to-answer latency. Background desktop activity is not controlled as in a dedicated benchmarking server. Full ranges are reported; repeated timing runs are not a hypothesis test.

The projected residual certificate is computed and recorded inside P-CONT. The separately recomputed terminal certificate is the shared original primal-dual gap, using the stored returned vector. For these ball-support penalties the shared gap is at most the projected residual certificate in exact arithmetic; this relation is derived in the manuscript. It is not asserted for every possible dual construction or regularizer.

Hardware is unchanged from the previous local campaign: Intel Core i7-12700H (14 cores, 20 logical processors), about 15.7 GiB RAM. [Hardware record](../paper_results/scale_review/hardware.json); Python/BLAS details are in the scale protocol and package versions are pinned in requirements.txt.
