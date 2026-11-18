# Person B guide, part 1: feature engineering

Builds on Part A's candidate tables (R=80 m, K=15). Every number below was produced by the scripts here, on the real data.

## 0. Files added or changed
| File | What it is |
|---|---|
| `src/features.py` | `FEATURES` list + `build_features(trip, cand)`: 25 features per (point, candidate) |
| `scripts/06_build_features.py` | builds `data/processed/features_{train,student_test}.parquet` + `artifacts/feature_list.json` |
| `scripts/07_feature_analysis.py` | feature-vs-label table, histograms, correlation, baseline check → `results/features/` |
| `tests/test_features.py` | synthetic two-way road: direction features separate twin arcs; features never use labels |
| `requirements.txt` | + `scikit-learn` |

## 1. Run it (on your laptop, from the repo root)
```powershell
pip install -r requirements.txt
cd scripts
python 03_build_candidates.py      # Part A output (parquet is not committed, so regenerate it)
python 06_build_features.py        # ~10 s
python 07_feature_analysis.py      # ~1 min
cd ..
python -m pytest -q tests          # 12 passed
```

## 2. Design rules (be ready to explain these)
- **No labels and no route in features.** Trips are loaded with `with_route=False`, the same path the API uses. `test_features_do_not_use_labels` proves it.
- **Same function for training and the API.** The API will call `build_features(trip, generate_candidates(...))`.
- **Metres, never degrees.** Everything uses Part A's `LocalProj` per trip.
- **The feature table is complete** (all 80 train trips). Suspect trips 33 and 54 are dropped at **training time** through `src.dataset.suspect_trips()`, the same rule `load_candidates()` uses.
- **Context features only look at neighbouring points of the same trip** (i±1, i±2, i±3), so there is no leakage across trips or folds.

## 3. The 25 features and why

| Group | Features | Why |
|---|---|---|
| Distance | `dist_m`, `rank`, `dist_minus_min`, `dist_ratio`, `n_cand_in_radius`, `n_cand_kept` | GPS noise is ~5–15 m, but the nearest arc is the true one only ~55% of the time. Relative closeness and ambiguity matter more than raw distance |
| Direction | `heading_diff_short` (i−1→i+1), `heading_diff_long` (i−3→i+3), `along_progress_m`, `stationary` | 95% of arcs have a reverse twin at the *same* distance. Only the travel direction separates them. `along_progress_m` = (projection of point i+1 − projection of point i−1) × arc length: positive on the right twin, negative on the reverse one |
| Kinematics | `speed_mps`, `dt_prev`, `dt_next` | Heading is unreliable when slow (`stationary` flags under 3 m of movement). Gaps make neighbours less informative |
| Geometry | `arc_len_m`, `proj_frac`, `outside_m` | A projection that falls past the end of the arc (`outside_m` > 0) usually means the next arc is correct |
| Topology | `to_out_deg`, `from_in_deg`, `two_way` | Junction complexity, and whether a twin exists |
| Context | `ctx_dist_prev/next`, `ctx_dist_prev3/next3`, `ctx_mean_dist_w5`, `nearest_vote_w7` | A car stays on an arc for several seconds. If the neighbouring points are also close to this arc, it is much more likely to be the right one. This smooths single noisy points before the HMM |

## 4. Results (`results/features/`)
**Univariate separation** (train, suspect trips excluded, AUC of each feature alone, `feature_summary.csv`):
`dist_minus_min` 0.936 · `rank` 0.933 · `dist_ratio` 0.929 · `ctx_mean_dist_w5` 0.905 · `dist_m` 0.899 · `ctx_dist_next/prev` 0.88 · `outside_m` 0.868 · `heading_diff_long` 0.842 · `heading_diff_short` 0.798 · `along_progress_m` 0.787 · `nearest_vote_w7` 0.759 · `arc_len_m` 0.730.
The weak ones alone (speed, dt, stationary, degrees, ~0.50–0.54) are **modifiers**: they tell the model *when* to trust heading or context. Tree models use them through interactions.

**Baseline check** (`baseline.csv`, GroupKFold(5) by trip, seed 42; Top-1 counts every GPS point, misses included):

| | Nearest-arc rule | Logistic regression on 25 features | LogReg macro-F1 |
|---|---|---|---|
| CV mean (78 train trips) | 55.4% | **88.4%** | 0.926 |
| Student test (10 trips) | 53.9% | **87.4%** | 0.920 |

So the features add about +33 pp over "pick the nearest road", before any non-linear model or HMM.

Figures for the report: `feature_histograms.png` (true arc vs others) and `feature_correlation.png`. The rank/distance features are strongly correlated; tree models handle that, but say so for Naive Bayes, which assumes independence.

## 5. Commits (from your own account, one at a time)
| # | Files | Message |
|---|---|---|
| 1 | `src/features.py`, `requirements.txt` | `feat: candidate feature engineering (25 features)` |
| 2 | `tests/test_features.py` | `test: direction/context features on synthetic two-way road` |
| 3 | `scripts/06_build_features.py`, `artifacts/feature_list.json` | `feat: build feature tables` |
| 4 | `scripts/07_feature_analysis.py`, `results/features/` | `eda: feature analysis and nearest-arc vs logreg baseline` |

Never commit `*.parquet` (already git-ignored).

## 6. Handoff to model training (next step)
- Data: `pd.read_parquet("data/processed/features_train.parquet")`, then drop `trip_id in suspect_trips()`.
- `X = df[FEATURES]`, `y = df.label`, `groups = df.trip_id` → `GroupKFold(5)`, seed 42.
- Report candidate macro-F1, log loss, and **point Top-1** (highest probability per point vs `points_*.parquet` `ref_arc`; points without candidates count as wrong). See `top1()` in `07_feature_analysis.py`.
- Don't use class weights if the probabilities feed the HMM (they distort calibration).

## 7. Viva questions for this part
1. Why isn't `dist_m` enough? (nearest arc = 55% Top-1; twins and parallel roads)
2. How does `along_progress_m` tell the two directions of a road apart? (sign of movement along the arc)
3. Why set heading to 90 when stationary? (direction is undefined, so 90 is the neutral midpoint between aligned and opposite)
4. Do context features leak labels? (no: GPS of neighbouring points only; there's a test)
5. Why do weak univariate features still help? (interactions: speed decides whether heading is trustworthy)
6. Why compute features for suspect trips at all? (complete table; the filter is a training decision)
