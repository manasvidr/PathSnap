"""Choose candidate radius R and cap K on TRAIN trips only: candidate miss rate vs average candidates per point.
-> results/radius_sweep_by_trip.csv, results/radius_sweep.csv"""
import time, numpy as np, pandas as pd
from _common import RESULTS
from src.io_utils import list_trips, load_trip
from src.candidates import build_index, generate_candidates
from src.labels import label_points, reference_quality

RS, KS = [30, 50, 80, 100], [10, 15, 20, None]
rows, t0 = [], time.time()
for name in list_trips("train"):
    trip = load_trip("train", name)
    lab = label_points(trip); q = reference_quality(lab)
    cand = generate_candidates(trip, radius_m=max(RS), k=None, index=build_index(trip))
    N = len(trip["track"])
    ref = lab.set_index("point_idx").ref_arc
    cand["is_ref"] = cand.arc_id.values == ref.reindex(cand.point_idx.values).values
    hit = cand[cand.is_ref].set_index("point_idx")
    d_ref = hit.dist_m.reindex(range(N)).values; r_ref = hit["rank"].reindex(range(N)).values
    for R in RS:
        cnt = np.bincount(cand.point_idx.values[cand.dist_m.values <= R], minlength=N)
        for K in KS:
            kk = K if K else 10**6
            miss = np.isnan(d_ref) | (d_ref > R) | (r_ref >= kk)
            rows.append(dict(trip_id=trip["trip_id"], ref_quality=q["ref_quality"], R=R, K=K or "all", n_points=N,
                             n_miss=int(miss.sum()), n_cand=int(np.minimum(cnt, kk).sum()), n_zero=int((cnt == 0).sum())))
bt = pd.DataFrame(rows); bt.to_csv(f"{RESULTS}/radius_sweep_by_trip.csv", index=False)
def agg(g):
    return pd.Series(dict(trips=g.trip_id.nunique(), points=g.n_points.sum(), miss_rate_pct=100 * g.n_miss.sum() / g.n_points.sum(),
                          avg_cand_per_point=g.n_cand.sum() / g.n_points.sum(), zero_cand_pct=100 * g.n_zero.sum() / g.n_points.sum(),
                          positive_frac=(g.n_points.sum() - g.n_miss.sum()) / g.n_cand.sum(), rows=g.n_cand.sum()))
all_ = bt.groupby(["R", "K"]).apply(agg, include_groups=False).assign(subset="all train")
ok = bt[bt.ref_quality == "ok"].groupby(["R", "K"]).apply(agg, include_groups=False).assign(subset="excl. suspect refs")
out = pd.concat([all_, ok]).reset_index(); out.to_csv(f"{RESULTS}/radius_sweep.csv", index=False)
pd.set_option("display.width", 200)
print(f"{time.time()-t0:.0f}s"); print(out.round(3).to_string(index=False))
