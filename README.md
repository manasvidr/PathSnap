# PathSnap

**GPS Map Matching Using Machine Learning**

PathSnap is a map-matching project developed for the UE24CS352A hackathon. Its goal is to match noisy GPS observations to road-network segments and reconstruct a vehicle's likely route.

## Project Overview

The project includes dataset exploration, road-network processing, candidate segment generation, reference-route alignment, visualization, and testing.

The repository is organized into separate modules so that different parts of the map-matching pipeline can be developed and evaluated independently.

## Project Structure

- `src/` — Data processing, geometry, candidate generation, labeling, and quality checks
- `scripts/` — Dataset inventory, candidate generation, and visualization scripts
- `tests/` — Automated tests
- `notebooks/` — Jupyter notebooks
- `results/` — Experiment and analysis outputs
- `artifacts/` — Generated project artifacts
- `docs/` — Technical documentation and final report

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/manasvidr/PathSnap.git
cd PathSnap
```

**2. Create a Python virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

## Dataset Setup

Place the provided training and student-test datasets inside:

```text
data/
  raw/
    train/
    student_test/
```

The raw datasets are excluded from Git tracking.

## Running the Data Processing Pipeline

From the repository root, execute:

```bash
python scripts/01_inventory.py
python scripts/02_radius_sweep.py
python scripts/03_build_candidates.py 80 15
python scripts/04_folium_maps.py
python scripts/05_eda_plots.py
```

These scripts support dataset inspection, candidate generation, and visualization.

## Running Tests

From the repository root:

```bash
python -m pytest -q tests
```

## Documentation

Additional information is available in:

- `docs/PERSON_A_GUIDE.md` — Dataset analysis and design decisions
- `docs/` — Project technical report
- `notebooks/` — Notebook-based exploration and experiments

## Team Members

| Name | SRN |
|---|---|
| Neeraj R Gowda | PES1UG24CS295 |
| Nayan Niranjana Kaushik | PES1UG24CS294 |
| Manasvi D R | PES1UG24CS262 |
| Manish Bala | PES1UG24CS263 |

## Note

The README describes the modules currently present in this repository. The complete project implementation should be checked against the final submission before running additional experiments.