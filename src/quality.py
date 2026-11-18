"""Per-trip inventory + data-quality metrics (one dict per trip) used by scripts/01_inventory.py."""
import os
import numpy as np
import networkx as nx
from .geo import haversine

SPEED_LIMIT = 70.0   # m/s, plan threshold for 'impossible' jumps
GAP_SPLIT_S = 10     # seconds; matches the HMM gap threshold proposed in the plan


def trip_quality(trip, root):
    tr, nd, ar = trip["track"], trip["nodes"], trip["arcs"]
    rt = trip["route"].arc_id.values
    r = dict(partition=trip["partition"], trip_id=trip["trip_id"], n_points=len(tr), n_nodes=len(nd),
             n_arcs=len(ar), n_route_arcs=len(rt))
    r["nan_cells"] = int(tr.isna().sum().sum() + nd.isna().sum().sum() + ar.isna().sum().sum())
    dt = np.diff(tr.t.values)
    d = haversine(tr.lon.values[:-1], tr.lat.values[:-1], tr.lon.values[1:], tr.lat.values[1:])
    sp = d / np.where(dt > 0, dt, np.nan)
    r.update(t_start=float(tr.t.iloc[0]), dup_ts=int((dt == 0).sum()), out_of_order=int((dt < 0).sum()),
             n_gaps_gt1s=int((dt > 1).sum()), n_gaps_gt10s=int((dt > GAP_SPLIT_S).sum()), max_gap_s=float(dt.max()),
             speed_max=float(np.nanmax(sp)), n_speed_gt70=int((sp > SPEED_LIMIT).sum()),
             track_len_km=float(d.sum() / 1000), identical_pos_steps=int((d == 0).sum()))
    r["coords_out_of_range"] = int(((tr.lon.abs() > 180) | (tr.lat.abs() > 90)).sum())
    r["lon_center"], r["lat_center"] = float(tr.lon.mean()), float(tr.lat.mean())
    r["extent_km_x"] = float(haversine(tr.lon.min(), tr.lat.mean(), tr.lon.max(), tr.lat.mean()) / 1000)
    r["extent_km_y"] = float(haversine(tr.lon.mean(), tr.lat.min(), tr.lon.mean(), tr.lat.max()) / 1000)
    r["track_in_node_bbox"] = bool(tr.lon.min() >= nd.lon.min() and tr.lon.max() <= nd.lon.max()
                                   and tr.lat.min() >= nd.lat.min() and tr.lat.max() <= nd.lat.max())
    r["arcs_bad_node_id"] = int(((ar.u < 0) | (ar.v < 0) | (ar.u >= len(nd)) | (ar.v >= len(nd))).sum())
    r["self_loops"], r["duplicate_arcs"] = int((ar.u == ar.v).sum()), int(ar.duplicated().sum())
    es = set(zip(ar.u, ar.v))
    r["pct_arcs_with_reverse"] = float(np.mean([(v, u) in es for u, v in es]))
    G = nx.DiGraph(); G.add_edges_from(es)
    r["isolated_nodes"] = len(nd) - G.number_of_nodes()   # nodes that appear in no arc
    comps = list(nx.weakly_connected_components(G))
    r["weak_components"] = len(comps)
    r["largest_component_frac"] = max(map(len, comps)) / G.number_of_nodes()
    r["route_arc_not_in_arcs"] = int(((rt < 0) | (rt >= len(ar))).sum())
    r["route_repeated_arcs"] = int(len(rt) - len(set(rt.tolist())))
    u, v = ar.u.values, ar.v.values
    r["route_disconnected_steps"] = int(sum(1 for a, b in zip(rt[:-1], rt[1:]) if a != b and v[a] != u[b]))
    tn = f"{trip['trip_id']:08d}"
    r["file_kb"] = sum(os.path.getsize(os.path.join(root, trip["partition"], tn, f"{tn}.{e}"))
                       for e in ("track", "nodes", "arcs", "route")) / 1024
    return r
