"""Feature tests on a synthetic network: a two-way road (arcs 0 east, 1 west) and a parallel one-way road 30 m north."""
import numpy as np
import pandas as pd

from src.candidates import build_index, generate_candidates
from src.features import FEATURES, build_features
from src.geo import LocalProj

LON0, LAT0 = 10.0, 50.0
M_LON = 1 / (111_195 * np.cos(np.radians(LAT0)))   # degrees per metre east
M_LAT = 1 / 111_195                                # degrees per metre north


def synthetic_trip():
    nodes = pd.DataFrame({"lon": [LON0, LON0 + 400 * M_LON, LON0, LON0 + 400 * M_LON],
                          "lat": [LAT0, LAT0, LAT0 + 30 * M_LAT, LAT0 + 30 * M_LAT]})
    arcs = pd.DataFrame({"u": [0, 1, 2], "v": [1, 0, 3]})          # 0: east, 1: west (twin), 2: parallel road east
    x = np.arange(50, 350, 10.0)                                    # car drives east at 10 m/s, 4 m north of road
    track = pd.DataFrame({"lon": LON0 + x * M_LON, "lat": LAT0 + 4 * M_LAT + 0 * x, "t": np.arange(len(x), dtype=float)})
    return dict(trip_id=1, partition="test", track=track, nodes=nodes, arcs=arcs, route=None)


def _features():
    trip = synthetic_trip()
    cand = generate_candidates(trip, 80.0, 15, build_index(trip))
    return cand, build_features(trip, cand, LocalProj.for_trip(trip))


def test_shape_and_finite():
    cand, f = _features()
    assert list(f.columns) == FEATURES and len(f) == len(cand)
    assert np.isfinite(f.values).all()


def test_direction_separates_twin_arcs():
    cand, f = _features()
    mid = cand.point_idx == 15
    east = f[mid & (cand.arc_id == 0)].iloc[0]
    west = f[mid & (cand.arc_id == 1)].iloc[0]
    assert east.dist_m == west.dist_m                       # distance alone cannot separate them
    assert east.heading_diff_short < 5 and west.heading_diff_short > 175
    assert east.along_progress_m > 15 and west.along_progress_m < -15


def test_context_prefers_road_the_car_is_on():
    cand, f = _features()
    mid = cand.point_idx == 15
    on_road = f[mid & (cand.arc_id == 0)].iloc[0]
    parallel = f[mid & (cand.arc_id == 2)].iloc[0]
    assert on_road.ctx_mean_dist_w5 < parallel.ctx_mean_dist_w5
    assert on_road.nearest_vote_w7 > parallel.nearest_vote_w7


def test_features_do_not_use_labels():
    trip = synthetic_trip()
    cand = generate_candidates(trip, 80.0, 15, build_index(trip))
    a = build_features(trip, cand.assign(label=0))
    b = build_features(trip, cand.assign(label=1))
    pd.testing.assert_frame_equal(a, b)
