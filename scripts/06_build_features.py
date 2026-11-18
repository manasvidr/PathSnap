"""Person B: build the feature tables from Part A's candidate tables.
-> data/processed/features_{partition}.parquet  (keys + label + FEATURES), artifacts/feature_list.json
Trips are loaded with with_route=False: features never see the reference route (same path as the API).
Suspect trips are NOT dropped here (the table is complete); filtering happens at training time via
src.dataset.suspect_trips(), exactly like load_candidates()."""
import json, os, time
import numpy as np, pandas as pd
from _common import ROOT, PROCESSED
from src.io_utils import PARTITIONS, load_trip
from src.features import FEATURES, build_features, network_tables
from src.geo import LocalProj

for part in PARTITIONS:
    t0 = time.time()
    cand = pd.read_parquet(f"{PROCESSED}/candidates_{part}.parquet")      # full table, no filtering
    out = []
    for tid, c in cand.groupby("trip_id", sort=True):
        trip = load_trip(part, f"{tid:08d}", with_route=False)
        c = c.reset_index(drop=True)
        proj = LocalProj.for_trip(trip)
        X = build_features(trip, c, proj, network_tables(trip, proj))
        out.append(pd.concat([c[["trip_id", "point_idx", "arc_id", "label"]], X], axis=1))
    F = pd.concat(out, ignore_index=True)
    assert len(F) == len(cand), "feature rows must match candidate rows"
    bad = ~np.isfinite(F[FEATURES].values)
    assert not bad.any(), f"non-finite values in {np.array(FEATURES)[bad.any(0)]}"
    F.to_parquet(f"{PROCESSED}/features_{part}.parquet", index=False)
    print(f"{part}: {F.shape[0]:,} rows x {len(FEATURES)} features, {time.time() - t0:.0f}s")

os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
json.dump({"features": FEATURES}, open(os.path.join(ROOT, "artifacts", "feature_list.json"), "w"), indent=2)
