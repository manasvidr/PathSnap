"""Candidate generation: all arcs within R metres of each GPS point (STRtree), capped at the K nearest."""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from .geo import LocalProj, arc_xy, point_segment_dist


def build_index(trip, proj=None):
    proj = proj or LocalProj.for_trip(trip)
    ax, ay, bx, by = arc_xy(trip, proj)
    segs = shapely.linestrings(np.stack([np.stack([ax, ay], 1), np.stack([bx, by], 1)], 1))
    return dict(proj=proj, ax=ax, ay=ay, bx=bx, by=by, tree=STRtree(segs))


def generate_candidates(trip, radius_m=80.0, k=15, index=None):
    """One row per (point, candidate arc). k=None keeps every arc in the radius.
    Columns: trip_id, point_idx, arc_id, dist_m, proj_frac, rank (0 = nearest), n_cand_in_radius (uncapped count).
    Points with no arc in the radius produce no rows (they count as candidate misses)."""
    idx = index or build_index(trip)
    px, py = idx["proj"].xy(trip["track"].lon.values, trip["track"].lat.values)
    pi, ai = idx["tree"].query(shapely.points(px, py), predicate="dwithin", distance=radius_m)
    d, t = point_segment_dist(px[pi], py[pi], idx["ax"][ai], idx["ay"][ai], idx["bx"][ai], idx["by"][ai])
    df = pd.DataFrame(dict(point_idx=pi.astype(np.int32), arc_id=ai.astype(np.int64),
                           dist_m=d.astype(np.float32), proj_frac=t.astype(np.float32)))
    df = df[df.dist_m <= radius_m].sort_values(["point_idx", "dist_m", "arc_id"], kind="mergesort")
    g = df.groupby("point_idx", sort=False)
    df["rank"] = g.cumcount().astype(np.int16)
    df["n_cand_in_radius"] = g["arc_id"].transform("size").astype(np.int16)
    if k is not None:
        df = df[df["rank"] < k]
    df.insert(0, "trip_id", np.int32(trip["trip_id"]))
    return df.reset_index(drop=True)
