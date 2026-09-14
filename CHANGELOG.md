# Changelog

## 2.1.0 — 2026-09-14

- Add 165 larger-scale main measurements and 12 separate safeguard ablations, with all failures preserved.
- Add matrix-free proximal Newton, projected recovery, residual-stage continuation, and bounded trial safeguards.
- Publish independent terminal audits, three new figure groups, operation/memory/time ranges, pilot records and theorem checks.
- Document the squared projected-stage error and bounded-trial work guarantee, with explicit limits and no general speed claim.
- Preserve v1/v2 artifacts and their recorded negative results.

## 2.0.0 — 2026-09-14

- Add common-original-gap continuation, direct smoothing, PQN, R-FISTA and PN experiments, with nonoverlapping group penalties and scale/conditioning/regularization/accuracy tests.
- Retain all 192 measured records, including 21 unsuccessful terminations; add independent gap checks, operation counts, peak process memory and three figure groups.
- Remove unconditional dense allocation from the new limited-memory solver. Record residual checks and safeguards; no hidden dense fallback.
- Add primary-source comparison and theory/implementation audit. Outer superlinear rates remain conditional, and stage decay is not total complexity.
- Reproduce all 231 archived runs and preserve the original v1 results unchanged. Timing rankings are not assumed invariant.
- Document that no general speed advantage is established; matrix-free PN and stochastic comparisons remain pending.
