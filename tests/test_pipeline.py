import numpy as np, pandas as pd
from src.candidates import generate_candidates
from src.labels import label_points
from src.graph import build_digraph

LAT = 50.0


def toy_trip(route):
    # nodes 0,1,2 on a line ~71 m apart; arcs: 0:0->1, 1:1->2, 2:2->1, 3:1->0
    nodes = pd.DataFrame(dict(lon=[10.0, 10.001, 10.002], lat=[LAT] * 3))
    arcs = pd.DataFrame(dict(u=[0, 1, 2, 1], v=[1, 2, 1, 0]))
    lon = [10.0002, 10.0008, 10.0012, 10.0018]
    track = pd.DataFrame(dict(lon=lon, lat=[LAT + 0.00002] * 4, t=[1.0, 2.0, 3.0, 4.0]))
    return dict(trip_id=1, partition="train", track=track, nodes=nodes, arcs=arcs, route=pd.DataFrame(dict(arc_id=route)))


def test_labels_follow_route_order():
    lab = label_points(toy_trip([0, 1]))
    assert lab.ref_arc.tolist() == [0, 0, 1, 1]
    assert lab.ref_dist_m.max() < 5


def test_labels_handle_revisited_arc():
    # route goes arc 0, arc 1, arc 0 again; points only cover the first two positions
    lab = label_points(toy_trip([0, 1, 0]))
    assert lab.route_pos.is_monotonic_increasing


def test_candidates_contain_true_arc_and_are_sorted():
    trip = toy_trip([0, 1])
    c = generate_candidates(trip, radius_m=80, k=15)
    assert set(c.columns) >= {"trip_id", "point_idx", "arc_id", "dist_m", "proj_frac", "rank", "n_cand_in_radius"}
    first = c[c.point_idx == 0]
    assert first.iloc[0].arc_id in (0, 3)          # nearest arcs share the same geometry (0->1 and its reverse 1->0)
    assert (c.groupby("point_idx")["dist_m"].apply(lambda s: s.is_monotonic_increasing)).all()
    assert (c["rank"] < 15).all()


def test_candidate_cap():
    c = generate_candidates(toy_trip([0, 1]), radius_m=80, k=1)
    assert c.groupby("point_idx").size().max() == 1


def test_digraph_weights():
    G = build_digraph(toy_trip([0, 1]))
    assert G.has_edge(0, 1) and 60 < G[0][1]["weight"] < 80
