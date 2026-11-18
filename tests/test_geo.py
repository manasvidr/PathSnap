import numpy as np
from src.geo import haversine, LocalProj, point_segment_dist


def test_haversine_one_degree_lat():
    assert abs(haversine(0, 0, 0, 1) - 111195) < 100


def test_projection_matches_haversine_over_25km():
    for lat in (0, 49.8, 56.2, 60):
        p = LocalProj(10.0, lat)
        x, y = p.xy(10.0 + 0.2, lat + 0.15)       # ~ 13 km E, 17 km N
        proj_d = np.hypot(x, y)
        true_d = haversine(10.0, lat, 10.2, lat + 0.15)
        assert abs(proj_d - true_d) / true_d < 0.005


def test_point_segment_dist():
    d, t = point_segment_dist(np.array([5., -3.]), np.array([4., 0.]), 0., 0., 10., 0.)
    assert np.allclose(d, [4., 3.]) and np.allclose(t, [0.5, 0.])
