"""Raw-file loading. Files have no headers and are whitespace separated (CRLF in .track)."""
import os
import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get("PATHSNAP_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
PARTITIONS = ("train", "student_test")  # evaluator trips are never on disk


def list_trips(partition, root=DATA_ROOT):
    """Sorted trip folder names (e.g. '00000001') of a partition."""
    return sorted(d for d in os.listdir(os.path.join(root, partition))
                  if os.path.isdir(os.path.join(root, partition, d)))


def _read(path, names, dtype=None):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(partition, trip_name, root=DATA_ROOT, with_route=True):
    """Return dict(trip_id, partition, track, nodes, arcs, route).
    track: lon, lat, t | nodes: lon, lat (row = node id) | arcs: u, v (row = arc id)
    route: DataFrame(arc_id) - reference only; pass with_route=False to model the hidden-label setting."""
    d = os.path.join(root, partition, trip_name)
    out = dict(
        trip_id=int(trip_name), partition=partition,
        track=_read(os.path.join(d, f"{trip_name}.track"), ["lon", "lat", "t"]),
        nodes=_read(os.path.join(d, f"{trip_name}.nodes"), ["lon", "lat"]),
        arcs=_read(os.path.join(d, f"{trip_name}.arcs"), ["u", "v"], dtype=np.int64),
        route=None,
    )
    rp = os.path.join(d, f"{trip_name}.route")
    if with_route and os.path.exists(rp):
        out["route"] = _read(rp, ["arc_id"], dtype=np.int64)
    return out
