"""Person B: feature sanity checks + evidence for the report ("what features and why").
Partitions: TRAIN (suspect trips dropped, as decided by the team) for all analysis and the CV baseline;
STUDENT_TEST only for the final baseline check. Outputs -> results/features/
  feature_summary.csv      mean of each feature for true vs wrong candidates + univariate AUC
  feature_histograms.png   positive vs negative distributions of the key features
  feature_correlation.png  correlation matrix (redundancy check)
  baseline.csv             nearest-arc rule vs logistic regression (GroupKFold(5) by trip, seed 42)"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from _common import RESULTS, PROCESSED, SEED
from src.dataset import suspect_trips, load_points
from src.features import FEATURES

OUT = os.path.join(RESULTS, "features"); os.makedirs(OUT, exist_ok=True)
tr = pd.read_parquet(f"{PROCESSED}/features_train.parquet")
tr = tr[~tr.trip_id.isin(suspect_trips())].reset_index(drop=True)
te = pd.read_parquet(f"{PROCESSED}/features_student_test.parquet")
pts_tr, pts_te = load_points("train"), load_points("student_test")
print(f"train {len(tr):,} rows / {tr.trip_id.nunique()} trips, student_test {len(te):,} rows")

# 1. per-feature separation (train)
rows = []
for c in FEATURES:
    auc = roc_auc_score(tr.label, tr[c])
    rows.append(dict(feature=c, mean_true_arc=tr.loc[tr.label == 1, c].mean(), mean_other=tr.loc[tr.label == 0, c].mean(),
                     univariate_auc=max(auc, 1 - auc), direction="higher=more likely" if auc >= .5 else "lower=more likely"))
S = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)
S.to_csv(os.path.join(OUT, "feature_summary.csv"), index=False)
print(S.round(3).to_string(index=False))

# 2. histograms of the key features
key = ["dist_m", "dist_minus_min", "heading_diff_short", "along_progress_m", "ctx_mean_dist_w5", "nearest_vote_w7", "speed_mps", "outside_m"]
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
sub = tr.sample(min(len(tr), 400_000), random_state=SEED)
for ax, c in zip(axes.ravel(), key):
    lo, hi = np.percentile(sub[c], [0.5, 99.5])
    bins = np.linspace(lo, hi if hi > lo else lo + 1, 50)
    ax.hist(sub.loc[sub.label == 0, c], bins=bins, density=True, alpha=.55, color="#9ca3af", label="other candidates")
    ax.hist(sub.loc[sub.label == 1, c], bins=bins, density=True, alpha=.65, color="#16a34a", label="true (reference) arc")
    ax.set_title(c); ax.grid(alpha=.3)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Feature distributions, true arc vs other candidates (train, suspect trips excluded)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "feature_histograms.png"), dpi=130); plt.close(fig)

# 3. correlation (redundancy)
C = sub[FEATURES].corr()
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(C.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=90, fontsize=7)
ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=7)
fig.colorbar(im, fraction=.046); ax.set_title("Feature correlation (train)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "feature_correlation.png"), dpi=130); plt.close(fig)


# 4. baselines: does the feature set beat "pick the nearest arc"?
def top1(rows, score, points):
    best = rows.assign(s=score).sort_values(["trip_id", "point_idx", "s"], ascending=[True, True, False]).drop_duplicates(["trip_id", "point_idx"])
    m = points.merge(best[["trip_id", "point_idx", "arc_id"]], on=["trip_id", "point_idx"], how="left")
    return float((m.arc_id.values == m.ref_arc.values).mean())   # points without candidates count as wrong


res = []
X, y, g = tr[FEATURES].values, tr.label.values, tr.trip_id.values
for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[a], y[a])
    p = m.predict_proba(X[b])[:, 1]
    vt = np.unique(g[b]); P = pts_tr[pts_tr.trip_id.isin(vt)]
    res.append(dict(split=f"cv_fold_{k}", nearest_arc_top1=top1(tr.iloc[b], -tr.dist_m.values[b], P),
                    logreg_top1=top1(tr.iloc[b], p, P), logreg_macro_f1=f1_score(y[b], p >= .5, average="macro")))
    print(res[-1])
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
p = m.predict_proba(te[FEATURES].values)[:, 1]
res.append(dict(split="student_test", nearest_arc_top1=top1(te, -te.dist_m.values, pts_te),
                logreg_top1=top1(te, p, pts_te), logreg_macro_f1=f1_score(te.label, p >= .5, average="macro")))
B = pd.DataFrame(res)
B.loc[len(B)] = dict(split="cv_mean", **B[B.split.str.startswith("cv")].drop(columns="split").mean().to_dict())
B.to_csv(os.path.join(OUT, "baseline.csv"), index=False)
print(B.round(4).to_string(index=False))
