# Measurement notes

The guard campaign was registered in formal/protocol.json before the first formal run. Four cases, six methods, three repeats, rotated method order. No case or run is excluded after observing results. Warm-up and measurement are separate; every method starts from zero. Raw terminal records include all continuation work and failed-trial work.

One benchmark worker runs at a time. This is a shared Windows desktop, not an isolated benchmarking host. Manuscript editing and one LaTeX compilation occurred while the guard batch ran. Ranges and deterministic operation counts are reported; small differences are not treated as speedup evidence. No measurements are rerun or selected to make a method win. The follow-up FBE batch will be kept separate and includes freshly measured P-CONT-D and R-FISTA controls.

A new FBE baseline follows the published fixed-envelope Algorithm 2, uses gamma=.95/L and memory ten, and computes the envelope gradient with exact logistic Hessian-vector products. Curvature pairs are formed between x and the accepted envelope line-search point w, before the FB update. It is our implementation, not an author-code reproduction. No tuning on the formal cases is performed.
