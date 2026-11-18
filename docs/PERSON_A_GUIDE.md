# Person A guide: data, EDA, candidates, labels, maps

Everything below was checked against the real files (90 trips: 80 train + 10 student_test). Where the plan PDF and the data disagree, the data wins and the row is marked **Δ**.

## 0. Setup (10 min)
```bash
git clone <team-repo> pathsnap && cd pathsnap
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data/raw && cp -r "<Student folder>/train" "<Student folder>/student_test" data/raw/   # data/raw is git-ignored
python -m pytest -q tests
```
VS Code: select the `.venv` interpreter, install the Python + Jupyter extensions. Scripts run from `scripts/` (`cd scripts && python 01_inventory.py`).

## 1. Ground rules
- Never commit `data/raw/`, `data/processed/` or any `*.parquet` (git-ignored). Commit the code that regenerates them.
- Evaluator trips (IDs 15, 17, 23, 30, 49, 59, 67, 73, 87, 94) are not in the zip. Never look for them; never put an ID list in code. The API reads trip IDs from the folder.
- Fix every choice (R, K, label rule, suspect threshold) using **train only**. Student-test is for comparison, not for tuning.
- Seed 42, GroupKFold(5) by `trip_id`. Candidate rows of one trip never cross folds.
- AI use must be disclosed and you must be able to explain every line in the viva. Read the code in `src/` before you commit it.

## 2. What the data actually looks like (plan vs reality)
| Topic | Plan says | Data says |
|---|---|---|
| Partitions | 80 + 10 (+10 hidden) | ✔ 80 / 10. 208,417 / 19,160 GPS points. Trip 1 matches the plan example exactly |
| Format | no headers, lon/lat | ✔ whitespace-separated, CRLF in `.track`, lon first |
| NaNs, bad coords, bad IDs | check | **none**: 0 NaN, 0 out-of-range, 0 out-of-bounds arc/route IDs, 0 isolated nodes |
| Timestamps | ~1 Hz | no duplicates or out-of-order. 3.3 % of steps >1 s (7,550); **617 gaps >10 s in 42 trips**; trips 90 and 78 have gaps of 153,064 s / 52,810 s; 16 trips start at t>1 (trip 6: 693 s). Student-test trips 65 (210 s gap) and 76 (206 gaps) need the HMM gap logic |
| Speed >70 m/s | impossible jumps | **Δ** 148 steps in 11 trips (trip 13: 95). These are GPS jumps = the noise you must match. Report, do not delete |
| Network | one-way vs both | median 95 % of arcs have a reverse arc, trip 80 only 35 %. Median 19 weak components/trip (largest holds ~98 % of nodes, trip 95: 69 %). 24 self-loops, 1,903 duplicate arcs, 106 zero-length arcs. Use `DiGraph` |
| Route connected? | check | **Δ** 14 disconnected steps in 11 trips (student-test: trip 76). Route repeats arcs in 10 trips (**student-test trip 56: 255 repeats**) |
| Labelling | nearest route arc, keep order forward | **Δ** needs monotone DP over route *positions*, otherwise revisits/loops break (see `src/labels.py`) |
| Reference quality | "reference route can be wrong" | **Δ** trips **54 and 33** (25k points): GPS sits on the network but the route is 120-140 m away (median). Flagged `suspect` |
| Projection | UTM via pyproj | **Δ** equirectangular per trip: trips span the globe (no single UTM zone), extents ≤ ~60 km, error <0.5 % (tested) |
| Class balance | ~1 positive per 10 rows | ✔ 9.6 % positives at R=80, K=15 |
| Nearest arc = label? | "strongest signal" | only **~50 %** (student-test 55 %). Parallel roads/junctions are the real problem, so heading + HMM matter |

## 3. Decisions made (all from train)
Sweep (`results/radius_sweep.csv`, 78 trips with a trustworthy reference):

| R (m) | K | miss % | avg cand/pt | rows |
|---|---|---|---|---|
| 50 | 15 | 1.51 | 7.6 | 1.40M |
| **80** | **15** | **0.92** | **9.8** | **1.80M** |
| 100 | 15 | 0.81 | 10.7 | 1.97M |
| 80 | all | 0.76 | 17.2 | 3.15M |

**Chosen: R = 80 m, K = 15.** Beyond that, miss rate gains are <0.15 pt for +10 % rows. Zero-candidate points: 0.28 % (R=80). Cost of the cap: trip 72 loses 11 % of points to K (dense network, true arc up to rank 37). Mention `n_cand_in_radius` to Person B (it is a column) and note K as a limitation.

With the two suspect trips included, miss rate is 11 % (all of it from trips 54/33). **Recommendation for team (needs an explicit team decision, then written in the report):** drop `ref_quality == "suspect"` trips from training/CV, keep them in EDA, and show their maps as an example of reference-route limitations. Student-test miss at R=80,K=15: 0.15 %.

## 4. Task order and commits (commit after each, push at least hourly)
| # | Task | File | Commit message |
|---|---|---|---|
| 1 | Loader, geo utils, tests | `src/io_utils.py`, `geo.py`, `tests/` | `feat: raw loaders and local projection` |
| 2 | Inventory + data quality | `src/quality.py`, `scripts/01_inventory.py` | `feat: inventory and data-quality table` |
| 3 | EDA plots | `scripts/05_eda_plots.py` | `eda: size, gap, trajectory and network plots` |
| 4 | Candidates (STRtree) | `src/candidates.py` | `feat: candidate generation` |
| 5 | Labels (monotone DP) + quality flag | `src/labels.py` | `feat: monotone route labelling + reference quality` |
| 6 | Radius/K sweep | `scripts/02_radius_sweep.py` | `exp: radius and K sweep on train` |
| 7 | Build parquet tables | `scripts/03_build_candidates.py` | `feat: labelled candidate tables` |
| 8 | Folium | `src/viz/folium_maps.py`, `scripts/04_folium_maps.py` | `feat: folium trip maps` |
| 9 | EDA notebook | `notebooks/eda.ipynb` | `eda: notebook` |
Branching: work on `feat/data-a`, open a PR early so B/C can pull `candidates` code. Tell B the moment `03_build_candidates.py` works: B needs `data/processed/candidates_*.parquet` (share by regenerating, not by committing).

## 5. Handoff contract
- `candidates_{train,student_test}.parquet`: `trip_id, point_idx, arc_id, dist_m, proj_frac, rank, n_cand_in_radius, label`.
- `points_*.parquet`: one row per GPS point: `ref_arc, ref_dist_m, route_pos, in_candidates`.
- `results/trip_reference_quality.csv`: `ref_quality` per trip. Use it to filter training.
- Inference path (Person D): `load_trip(..., with_route=False)` → `build_index` → `generate_candidates`. No labels anywhere in that path. Ground truth is only used server-side for metrics.
- Person C needs per-point times and the road graph: build the `DiGraph` from `trip["arcs"]` (arc id = row index) in their code.

## 6. Report text you can write from real numbers (partition always stated)
- *Dataset & EDA (train + student_test):* inventory table (`inventory.csv`), figures in `results/eda/`, missing-value result (none), gap table, speed outliers, connectivity table, reference-route limitations (disconnected steps, repeated arcs, suspect trips).
- *Preprocessing:* no rows are dropped from tracks; gaps are kept and handled by the HMM; local metric projection; labelling method + why; candidate radius choice with the sweep table.
- *Folium:* `results/maps/trip_00000001.html` (clean), `trip_00000054.html` (suspect reference), screenshot at least one.
- Always describe labels as "reference-derived labels", not ground truth.

## 7. Viva checklist
1. Why is a degree-based distance wrong? 2. Why equirectangular and what is its error? 3. What does each file contain and how do arc/node IDs map? 4. Walk through the DP labelling and why positions not arc IDs. 5. Why R=80/K=15, what does the cap cost? 6. Why are trips 54/33 suspect and what did the team decide? 7. What is candidate miss rate and why does it bound Top-1? 8. Which student-test trips will be hard (56 revisits, 65/76 gaps, 9 dense)?
