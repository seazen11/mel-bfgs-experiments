# Data and attribution

Six synthetic instances are generated with NumPy's default RNG, seeds 0, 1,
and 2, correlations 0 and 0.9, 600 samples, and 80 variables. The first ten
true coefficients are nonzero. Binary labels are sampled from a logistic
model. Features are standardized, and no intercept is fitted.

WDBC is downloaded automatically into the run directory. It has 569 rows
and 30 features. All rows are used in optimization; there is no train/test
split and no claim about prediction or generalization.

- Source: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
- DOI: https://doi.org/10.24432/C5DW2B
- License: CC BY 4.0 (the dataset is not covered by this repository's MIT license).
- Attribution: Wolberg, W., Mangasarian, O., Street, N., & Street, W.
  Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository.
- Download: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

To work offline, place `wdbc.data` in `<output>/data/` before running.
The loader checks SHA-256 before parsing. Expected SHA-256:

`d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`

The patient identifier column is discarded. The raw dataset is not bundled.
