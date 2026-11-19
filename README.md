# PathSnap

## GPS Map Matching Using Machine Learning

**Course:** UE24CS352A – Machine Learning Hackathon

PathSnap is a GPS map-matching system designed to match noisy GPS observations to the most likely road segments. The project combines machine learning, road-network geometry, and sequence-based decoding to improve route reconstruction.

## 1. Project Overview

GPS coordinates may contain noise, making it difficult to determine which road a vehicle is travelling on. PathSnap addresses this problem by identifying nearby candidate road segments, extracting relevant features, predicting the most likely segment, and applying road-network constraints to reconstruct a continuous route.

The complete project implementation includes dataset preprocessing, exploratory data analysis, candidate generation, model training, evaluation, and HMM/Viterbi decoding.

## 2. Dataset

The project uses GPS trajectory observations and road-network information.

The dataset is divided into:

- **Training:** 80 trips
- **Student test:** 10 trips
- **Reserved evaluator test:** 10 trips

The data includes:

- `.track` — GPS observations and timestamps
- `.nodes` — Road-network nodes
- `.arcs` — Directed road segments
- `.route` — Reference route information

The training data contains approximately 208,417 GPS observations.

The reserved evaluator test data is not used during model training.

## 3. Methodology

The main stages of the project are:

1. **Data preprocessing:** Load GPS trajectories and road-network information, validate the data, and handle quality issues.
2. **Exploratory data analysis:** Examine trajectory characteristics, GPS noise, and road-network properties.
3. **Candidate generation:** Identify nearby road segments for each GPS observation.
4. **Feature engineering:** Generate features describing GPS-to-road geometry and movement characteristics.
5. **Model training:** Train and compare supervised machine-learning classifiers.
6. **Model evaluation:** Evaluate models using trip-based cross-validation and student-test data.
7. **Sequence decoding:** Apply HMM/Viterbi decoding with road-network connectivity constraints.
8. **Visualization:** Display GPS trajectories, candidate roads, and predicted routes on interactive maps.

## 4. Machine Learning Models

Five classification models were evaluated:

- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- Histogram Gradient Boosting

### Model Comparison

| Model | Cross-Validation Macro F1 | Student-Test Top-1 Accuracy |
|---|---:|---:|
| Logistic Regression | 0.8089 | 79.65% |
| Gaussian Naive Bayes | 0.7372 | 77.15% |
| Decision Tree | 0.8464 | 79.18% |
| Random Forest | 0.8876 | 80.54% |
| Histogram Gradient Boosting | **0.8900** | **80.68%** |

Histogram Gradient Boosting was selected based on the highest cross-validation macro F1 score.

Optuna was also used for hyperparameter optimization.

## 5. HMM/Viterbi Decoding

The project evaluates a sequence-decoding approach to improve route continuity.

Viterbi decoding considers both model predictions and road-network connectivity.

In the recorded student-test comparison:

- Mean classifier pointwise accuracy: **83.14%**
- Mean decoded pointwise accuracy: **81.88%**
- Mean direct road-continuity score increased from approximately **89.90% to 94.76%**

The decoder improved continuity overall but did not improve pointwise accuracy on every trip.

## 6. Repository Structure

The repository includes the following directories:

```text
PathSnap/
├── src/             # Data processing and road-network utilities
├── scripts/         # Dataset processing and visualization scripts
├── tests/           # Automated tests
├── notebooks/       # Jupyter notebooks
├── docs/            # Documentation and technical report
├── results/         # Analysis and experiment results
├── artifacts/       # Generated artifacts
├── requirements.txt
└── README.md
```

The complete local implementation also contains additional ML pipeline, API, and visualization modules. The GitHub repository should be synchronized with the final implementation before submission.

## 7. Installation

### Clone the repository

```bash
git clone https://github.com/manasvidr/PathSnap.git
cd PathSnap
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 8. Dataset Setup

Place the provided datasets inside:

```text
data/
└── raw/
    ├── train/
    └── student_test/
```

Raw datasets should not be committed to GitHub unless their distribution is permitted.

## 9. Running the Available Data Pipeline

From the repository root, run:

```bash
python scripts/01_inventory.py
python scripts/02_radius_sweep.py
python scripts/03_build_candidates.py 80 15
python scripts/04_folium_maps.py
python scripts/05_eda_plots.py
```

These commands cover the repository's data inspection, candidate generation, and visualization scripts.

## 10. Running Tests

```bash
python -m pytest -q tests
```

## 11. Jupyter Notebook

The project notebook is available at:

[PathSnap Experiment Notebook](https://github.com/manasvidr/PathSnap/blob/main/notebooks/PathSnap_Experiment_Notebook.ipynb)

## 12. Documentation

The `docs/` folder contains supporting documentation, including dataset analysis and the final technical report.

## 13. Team Members

| Name | SRN |
|---|---|
| Neeraj R Gowda | PES1UG24CS295 |
| Nayan Niranjana Kaushik | PES1UG24CS294 |
| Manasvi D R | PES1UG24CS262 |
| Manish Bala | PES1UG24CS263 |

## 14. AI Tool Usage

ChatGPT was used as an AI-assisted tool for understanding concepts, coding assistance, debugging, and technical documentation. The project team is responsible for reviewing and validating the implementation and experimental results.

## 15. Conclusion

PathSnap demonstrates how machine-learning classification and road-network-aware sequence decoding can be combined for GPS map matching.

Histogram Gradient Boosting achieved the highest cross-validation macro F1 score among the evaluated classifiers. HMM/Viterbi decoding improved road continuity overall, although pointwise prediction accuracy decreased slightly on average.

The project highlights the importance of combining classification performance with road-network constraints when reconstructing vehicle trajectories.

## 16. Submission Note

The README describes the modules currently present in this repository. The complete project implementation should be checked against the final submission before running additional experiments.
