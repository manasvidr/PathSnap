"""Geometry. Per-trip local equirectangular projection (metres, centred on the track centroid).
Chosen over UTM: trips are in different parts of the world (no single UTM zone), extents are <= ~25 km,
and the scale error stays well under 1 % (checked against haversine in tests/test_geo.py)."""
import numpy as np

R_EARTH = 6371008.8


def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


class LocalProj:
    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = np.radians(1.0) * R_EARTH * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * R_EARTH

    @classmethod
    def for_trip(cls, trip):
        return cls(trip["track"].lon.mean(), trip["track"].lat.mean())

    def xy(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def lonlat(self, x, y):
        return self.lon0 + np.asarray(x) / self.kx, self.lat0 + np.asarray(y) / self.ky


def point_segment_dist(px, py, ax, ay, bx, by):
    """Vectorised distance from points to segments (all in metres). Returns (dist, t) with t in [0,1] = projection fraction."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def arc_xy(trip, proj):
    """Projected endpoints of every arc: (ax, ay, bx, by)."""
    X, Y = proj.xy(trip["nodes"].lon.values, trip["nodes"].lat.values)
    u, v = trip["arcs"].u.values, trip["arcs"].v.values
    return X[u], Y[u], X[v], Y[v]
