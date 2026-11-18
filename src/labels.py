"""Per-point labels from the reference route (which gives arcs in order, but NOT which point belongs to which arc).
Method: monotone alignment. Each GPS point i gets a route position j(i), non-decreasing in i, minimising
sum_i dist(point_i, route_arc[j(i)]) by DP (O(N*M)). Label arc = route_arc[j(i)].
Why not 'nearest route arc' per point: routes revisit streets (student-test trip 56 repeats 255 arcs) and loops/parallel
segments make the unconstrained nearest arc jump backwards in route order. We align route POSITIONS, so revisits work.
Labels are a design decision, not ground truth - see reference_quality()."""
import numpy as np
import pandas as pd
from .geo import LocalProj, arc_xy, point_segment_dist

SUSPECT_MEDIAN_M = 30.0  # median point-to-label distance above this => reference route inconsistent with the GPS


def label_points(trip, proj=None):
    proj = proj or LocalProj.for_trip(trip)
    ax, ay, bx, by = arc_xy(trip, proj)
    px, py = proj.xy(trip["track"].lon.values, trip["track"].lat.values)
    ra = trip["route"].arc_id.values
    N, M = len(px), len(ra)
    cost = np.empty((N, M), dtype=np.float32)
    for j, a in enumerate(ra):
        cost[:, j] = point_segment_dist(px, py, ax[a], ay[a], bx[a], by[a])[0]
    bp = np.zeros((N, M), dtype=np.int16 if M < 32000 else np.int32)
    pos = np.arange(M)
    dp = cost[0].copy()
    for i in range(1, N):
        cm = np.minimum.accumulate(dp)
        bp[i] = np.maximum.accumulate(np.where(dp == cm, pos, 0))
        dp = cost[i] + cm
    j = int(np.argmin(dp))
    lab = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        lab[i] = j
        j = int(bp[i, j])
    return pd.DataFrame(dict(trip_id=np.int32(trip["trip_id"]), point_idx=np.arange(N, dtype=np.int32),
                             route_pos=lab.astype(np.int32), ref_arc=ra[lab],
                             ref_dist_m=cost[np.arange(N), lab]))


def reference_quality(lab):
    med, p95 = float(lab.ref_dist_m.median()), float(lab.ref_dist_m.quantile(0.95))
    return dict(ref_dist_med=med, ref_dist_p95=p95, ref_quality="suspect" if med > SUSPECT_MEDIAN_M else "ok")
