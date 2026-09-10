# Provenance

`source/` preserves the three local experiment scripts before packaging.
The runnable copies in `experiments/` change output paths, remove the local
dependency-directory assumption, and add a WDBC checksum check. Numerical
updates, tolerances, seeds, repetition order, and plotting rules are unchanged.

Published timings were measured before packaging. They were not remeasured
to make the public copy appear faster. The expanded batch supersedes the
initial comparison timings; supporting records are from the earlier batch.
The discarded initial expanded logging comparison is not part of this release.
BLAS file paths in JSON metadata are redacted to library basenames. Values,
iteration histories, certificates, and times are retained.

Original source hashes remain as historical metadata; use `SHA256SUMS.json`
for the released files. The `source_sha256` inside the expanded protocol
refers to the timed runner before the final font and axis-only changes.
