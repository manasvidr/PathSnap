"""EDA figures from results/inventory.csv (+ raw tracks) -> results/eda/*.png. State partition in every title."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import RESULTS
from src.io_utils import list_trips, load_trip
from src.geo import LocalProj

inv = pd.read_csv(f"{RESULTS}/inventory.csv"); OUT = f"{RESULTS}/eda"
col = {"train": "#1f77b4", "student_test": "#ff7f0e"}

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, c, t in zip(ax, ["n_points", "n_arcs", "n_route_arcs"], ["GPS points / trip", "Road arcs / trip (log)", "Reference-route arcs / trip"]):
    for p, g in inv.groupby("partition"):
        a.hist(np.log10(g[c]) if c == "n_arcs" else g[c], bins=25, alpha=.6, color=col[p], label=p)
    a.set_title(t); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/hist_sizes.png", dpi=130); plt.close()

gaps, spd = [], []
for p in ["train", "student_test"]:
    for n in list_trips(p):
        tr = load_trip(p, n, with_route=False)["track"]; dt = np.diff(tr.t.values)
        gaps += [(p, g) for g in dt[dt > 1]]
g = pd.DataFrame(gaps, columns=["partition", "dt"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(np.log10(g.dt), bins=40, color="#1f77b4"); ax[0].set_xlabel("log10(time gap, s)"); ax[0].set_title(f"Gaps > 1 s (n={len(g)}), train + student_test")
ax[1].bar(["1 s", "1-10 s", "10-60 s", ">60 s"], [(g.dt <= 1).sum() + int(inv.n_points.sum() - len(inv) - len(g)), ((g.dt > 1) & (g.dt <= 10)).sum(), ((g.dt > 10) & (g.dt <= 60)).sum(), (g.dt > 60).sum()])
ax[1].set_yscale("log"); ax[1].set_title("Step intervals, all 90 trips")
plt.tight_layout(); plt.savefig(f"{OUT}/time_gaps.png", dpi=130); plt.close()

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
sel = [("train", 1), ("train", 13), ("train", 54), ("train", 33), ("student_test", 56), ("student_test", 76)]
for a, (p, t) in zip(ax.ravel(), sel):
    trip = load_trip(p, f"{t:08d}"); pr = LocalProj.for_trip(trip)
    x, y = pr.xy(trip["track"].lon, trip["track"].lat)
    a.scatter(x / 1000, y / 1000, s=1, c=trip["track"].t, cmap="viridis"); a.set_aspect("equal")
    a.set_title(f"{p} trip {t}: {len(x)} pts"); a.set_xlabel("km E"); a.set_ylabel("km N")
plt.tight_layout(); plt.savefig(f"{OUT}/trajectories.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(inv.pct_arcs_with_reverse, bins=20, color="#1f77b4"); ax[0].set_title("Share of arcs with a reverse arc (90 trips)")
ax[1].hist(inv.largest_component_frac, bins=20, color="#1f77b4"); ax[1].set_title("Largest weak component / nodes (90 trips)")
ax[2].scatter(inv.n_arcs, inv.n_points, c=inv.partition.map(col), s=14); ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("arcs"); ax[2].set_ylabel("GPS points"); ax[2].set_title("Network size vs track length")
plt.tight_layout(); plt.savefig(f"{OUT}/network.png", dpi=130); plt.close()

rq = pd.read_csv(f"{RESULTS}/trip_reference_quality.csv")
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.log10(rq.ref_dist_med.clip(lower=1)), bins=30); ax.axvline(np.log10(30), c="r", ls="--", label="30 m suspect threshold")
ax.set_xlabel("log10(median point-to-label distance, m)"); ax.set_title("Reference-route consistency, 90 trips"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/reference_quality.png", dpi=130); plt.close()
print("saved", sorted(__import__("os").listdir(OUT)))
