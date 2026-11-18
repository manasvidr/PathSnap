"""Inventory + data-quality table for every trip -> results/inventory.csv"""
import time, pandas as pd
from _common import RESULTS
from src.io_utils import DATA_ROOT, PARTITIONS, list_trips, load_trip
from src.quality import trip_quality

rows, t0 = [], time.time()
for part in PARTITIONS:
    for name in list_trips(part):
        rows.append(trip_quality(load_trip(part, name), DATA_ROOT))
df = pd.DataFrame(rows)
df.to_csv(f"{RESULTS}/inventory.csv", index=False)
print(f"{len(df)} trips in {time.time()-t0:.0f}s -> results/inventory.csv")
print(df.groupby("partition")[["n_points", "n_nodes", "n_arcs", "n_route_arcs"]].agg(["count", "sum", "median", "max"]).T.to_string())
