"""Single source of truth for Part A choices (all fixed from TRAIN-only analysis, see results/radius_sweep.csv).
The API (Person D) must import these so inference uses the same candidate settings as training."""
SEED = 42
CAND_RADIUS_M = 80.0       # candidate search radius
CAND_K = 15                # max candidates per point (K nearest)
SUSPECT_MEDIAN_M = 30.0    # trip is 'suspect' if median point-to-reference-arc distance exceeds this
GAP_SPLIT_S = 10           # time gap (s) treated as a break by the HMM / quality report
