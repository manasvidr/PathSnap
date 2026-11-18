"""Build labelled candidate tables for train and student_test.
-> data/processed/candidates_{partition}.parquet, points_{partition}.parquet, results/trip_reference_quality.csv
Candidate columns: trip_id, point_idx, arc_id, dist_m, proj_frac, rank, n_cand_in_radius, label (0/1).
points_*: one row per GPS point (ref_arc, ref_dist_m, in_candidates) -> candidate miss rate.
Labels exist only for building train/dev data; the API must call generate_candidates() WITHOUT labels."""
import sys, time, numpy as np, pandas as pd
from _common import RESULTS, PROCESSED
from src.io_utils import PARTITIONS, list_trips, load_trip
from src.candidates import build_index, generate_candidates
from src.labels import label_points, reference_quality

R = float(sys.argv[1]) if len(sys.argv) > 1 else 80.0
K = int(sys.argv[2]) if len(sys.argv) > 2 else 15
qrows = []
for part in PARTITIONS:
    C, P, t0 = [], [], time.time()
    for name in list_trips(part):
        trip = load_trip(part, name)
        lab = label_points(trip); q = reference_quality(lab)
        cand = generate_candidates(trip, R, K, build_index(trip))
        ref = lab.ref_arc.values
        cand["label"] = (cand.arc_id.values == ref[cand.point_idx.values]).astype(np.int8)
        found = np.zeros(len(lab), bool); found[cand.point_idx.values[cand.label.values == 1]] = True
        lab["in_candidates"] = found
        qrows.append(dict(partition=part, trip_id=trip["trip_id"], n_points=len(lab), R=R, K=K, **q,
                          miss_rate=1 - found.mean(), avg_cand=len(cand) / len(lab), n_rows=len(cand)))
        C.append(cand); P.append(lab)
    C, P = pd.concat(C, ignore_index=True), pd.concat(P, ignore_index=True)
    C.to_parquet(f"{PROCESSED}/candidates_{part}.parquet", index=False)
    P.to_parquet(f"{PROCESSED}/points_{part}.parquet", index=False)
    print(f"{part}: {len(C):,} rows, positives {C.label.mean():.3f}, miss {1-P.in_candidates.mean():.4f}, {time.time()-t0:.0f}s")
pd.DataFrame(qrows).to_csv(f"{RESULTS}/trip_reference_quality.csv", index=False)
