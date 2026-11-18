"""Feature engineering (Person B): one feature row per (GPS point, candidate arc) row of generate_candidates().

Inference-safe: every feature is computed from the GPS track and the road network only (.track/.nodes/.arcs).
Labels and the reference route are never used, so the API calls exactly the same function.

Feature groups and why they exist
---------------------------------
Distance      dist_m, rank, dist_minus_min, dist_ratio, n_cand_in_radius, n_cand_kept
              GPS noise is ~5-15 m, so the true arc is usually close - but only ~50 % of the time the
              *nearest* (Part A finding), so relative closeness and ambiguity (how many arcs nearby) matter.
Direction     heading_diff_short, heading_diff_long, along_progress_m, stationary
              95 % of arcs have a reverse twin at the SAME distance; parallel roads are close too.
              Only the direction of travel separates them. along_progress_m (signed movement along the arc
              between point i-1 and i+1) is positive on the correct twin and negative on the reverse one.
Kinematics    speed_mps, dt_prev, dt_next
              heading is unreliable when slow; gaps (617 > 10 s in train) make neighbours less informative.
Geometry      arc_len_m, proj_frac, outside_m
              short connector arcs at junctions vs long road segments; projections falling past the arc end
              usually mean "the next arc" is the right one.
Topology      to_out_deg, from_in_deg, two_way
              junction complexity and whether a reverse twin exists.
Context       ctx_dist_prev, ctx_dist_next, ctx_dist_prev3, ctx_dist_next3, ctx_mean_dist_w5, nearest_vote_w7
              a car stays on an arc for several seconds: if neighbouring GPS points are also close to this
              arc it is far more likely. This smooths single noisy points before the HMM.
"""
import numpy as np
import pandas as pd

from .geo import LocalProj, arc_xy

FEATURES = [
    "dist_m", "rank", "dist_minus_min", "dist_ratio", "n_cand_in_radius", "n_cand_kept",
    "heading_diff_short", "heading_diff_long", "along_progress_m", "stationary",
    "speed_mps", "dt_prev", "dt_next",
    "arc_len_m", "proj_frac", "outside_m",
    "to_out_deg", "from_in_deg", "two_way",
    "ctx_dist_prev", "ctx_dist_next", "ctx_dist_prev3", "ctx_dist_next3", "ctx_mean_dist_w5", "nearest_vote_w7",
]

STATIONARY_M = 3.0      # displacement between i-1 and i+1 below this => heading undefined
CTX_CLIP_M = 200.0      # cap context distances (a far neighbour carries no extra information)
HEADING_NEUTRAL = 90.0  # value used when heading is undefined (halfway between aligned and opposite)


def _bearing(dx, dy):
    """Compass bearing in degrees [0, 360) of a displacement (x east, y north)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def _angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def _seg_t_raw(px, py, ax, ay, bx, by):
    """Distance and UNclipped projection fraction (t < 0 or t > 1 = projection outside the arc)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0), 0.0)
    tc = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + tc * dx), py - (ay + tc * dy)), t


def _shift(n, o):
    return np.clip(np.arange(n) + o, 0, n - 1)


def network_tables(trip, proj):
    """Per-arc geometry and topology for the whole network (arc id = row index, as in Part A)."""
    ax, ay, bx, by = arc_xy(trip, proj)
    u, v = trip["arcs"].u.values, trip["arcs"].v.values
    n_nodes = len(trip["nodes"])
    real = u != v                                          # self-loops carry no movement
    pairs = pd.DataFrame({"u": u[real], "v": v[real]}).drop_duplicates()   # duplicate arcs counted once
    out_deg = np.bincount(pairs.u.values, minlength=n_nodes)
    in_deg = np.bincount(pairs.v.values, minlength=n_nodes)
    key = pd.MultiIndex.from_arrays([u, v])
    rev = pd.MultiIndex.from_arrays([v, u])
    two_way = rev.isin(key) & real
    return dict(ax=ax, ay=ay, bx=bx, by=by, u=u, v=v, length=np.hypot(bx - ax, by - ay),
                bearing=_bearing(bx - ax, by - ay), out_deg=out_deg, in_deg=in_deg, two_way=two_way)


def point_kinematics(px, py, t):
    n = len(px)
    m1, p1, m3, p3 = _shift(n, -1), _shift(n, 1), _shift(n, -3), _shift(n, 3)
    dxs, dys = px[p1] - px[m1], py[p1] - py[m1]
    dxl, dyl = px[p3] - px[m3], py[p3] - py[m3]
    disp_s, disp_l = np.hypot(dxs, dys), np.hypot(dxl, dyl)
    dts = np.maximum(t[p1] - t[m1], 1e-6)
    return dict(head_s=_bearing(dxs, dys), head_l=_bearing(dxl, dyl), disp_s=disp_s, disp_l=disp_l,
                speed=disp_s / dts, dt_prev=np.r_[0.0, np.diff(t)], dt_next=np.r_[np.diff(t), 0.0])


def build_features(trip, cand, proj=None, net=None):
    """trip: dict from load_trip (route not needed). cand: rows from generate_candidates for this trip.
    Returns a float32 DataFrame with FEATURES, index aligned with cand."""
    proj = proj or LocalProj.for_trip(trip)
    net = net or network_tables(trip, proj)
    tr = trip["track"]
    px, py = proj.xy(tr.lon.values, tr.lat.values)
    t = tr.t.values.astype(float)
    n = len(px)
    pi = cand.point_idx.values
    a = cand.arc_id.values
    ax, ay, bx, by, L = net["ax"][a], net["ay"][a], net["bx"][a], net["by"][a], net["length"][a]
    kin = point_kinematics(px, py, t)
    f = pd.DataFrame(index=cand.index)

    # distance
    d = cand.dist_m.values.astype(float)
    g = cand.groupby("point_idx", sort=False).dist_m
    dmin = g.transform("min").values.astype(float)
    f["dist_m"] = d
    f["rank"] = cand["rank"].values
    f["dist_minus_min"] = d - dmin
    f["dist_ratio"] = (d + 1.0) / (dmin + 1.0)
    f["n_cand_in_radius"] = cand.n_cand_in_radius.values
    f["n_cand_kept"] = g.transform("size").values

    # direction
    stat = kin["disp_s"][pi] < STATIONARY_M
    f["heading_diff_short"] = np.where(stat, HEADING_NEUTRAL, _angle_diff(kin["head_s"][pi], net["bearing"][a]))
    f["heading_diff_long"] = np.where(kin["disp_l"][pi] < STATIONARY_M, HEADING_NEUTRAL,
                                      _angle_diff(kin["head_l"][pi], net["bearing"][a]))
    _, t_prev = _seg_t_raw(px[_shift(n, -1)[pi]], py[_shift(n, -1)[pi]], ax, ay, bx, by)
    _, t_next = _seg_t_raw(px[_shift(n, 1)[pi]], py[_shift(n, 1)[pi]], ax, ay, bx, by)
    f["along_progress_m"] = np.clip((t_next - t_prev) * L, -100, 100)
    f["stationary"] = stat.astype(np.int8)

    # kinematics
    f["speed_mps"] = np.clip(kin["speed"][pi], 0, 80)
    f["dt_prev"] = np.clip(kin["dt_prev"][pi], 0, 120)
    f["dt_next"] = np.clip(kin["dt_next"][pi], 0, 120)

    # geometry
    _, t_raw = _seg_t_raw(px[pi], py[pi], ax, ay, bx, by)
    f["arc_len_m"] = L
    f["proj_frac"] = cand.proj_frac.values
    f["outside_m"] = np.maximum(0, np.maximum(-t_raw, t_raw - 1)) * L

    # topology
    f["to_out_deg"] = net["out_deg"][net["v"][a]]
    f["from_in_deg"] = net["in_deg"][net["u"][a]]
    f["two_way"] = net["two_way"][a].astype(np.int8)

    # temporal context: distance of neighbouring observations to THIS arc
    cd = {}
    for o in (-3, -2, -1, 1, 2, 3):
        j = _shift(n, o)[pi]
        cd[o], _ = _seg_t_raw(px[j], py[j], ax, ay, bx, by)
    f["ctx_dist_prev"] = np.minimum(cd[-1], CTX_CLIP_M)
    f["ctx_dist_next"] = np.minimum(cd[1], CTX_CLIP_M)
    f["ctx_dist_prev3"] = np.minimum(cd[-3], CTX_CLIP_M)
    f["ctx_dist_next3"] = np.minimum(cd[3], CTX_CLIP_M)
    f["ctx_mean_dist_w5"] = np.minimum((cd[-2] + cd[-1] + d + cd[1] + cd[2]) / 5.0, CTX_CLIP_M)

    # share of the 7 surrounding points whose nearest candidate is this arc
    nearest = np.full(n, -1, dtype=np.int64)
    top = cand.loc[cand.groupby("point_idx", sort=False).dist_m.idxmin()]
    nearest[top.point_idx.values] = top.arc_id.values
    votes = np.zeros(len(cand))
    for o in range(-3, 4):
        votes += nearest[_shift(n, o)[pi]] == a
    f["nearest_vote_w7"] = votes / 7.0
    return f[FEATURES].astype(np.float32)
