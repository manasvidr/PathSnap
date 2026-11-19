\# PathSnap — Testing and Execution Guide



\## 1. Purpose



This document explains how to set up, execute, and test the PathSnap GPS map-matching project.



\## 2. Prerequisites



\- Python 3

\- Git

\- Project dependencies listed in `requirements.txt`

\- Training and student-test datasets



\## 3. Environment Setup



Clone the repository:



```bash

git clone https://github.com/manasvidr/PathSnap.git

cd PathSnap

```



Create a virtual environment:



```bash

python -m venv .venv

```



Activate it on Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

python -m pip install -r requirements.txt

```



\## 4. Dataset Preparation



Place the provided datasets in the following directory structure:



```text

data/

└── raw/

&#x20;   ├── train/

&#x20;   └── student\_test/

```



Raw datasets are excluded from version control.



\## 5. Running the Data Processing Scripts



Run the following commands from the repository root:



```bash

python scripts/01\_inventory.py

python scripts/02\_radius\_sweep.py

python scripts/03\_build\_candidates.py 80 15

python scripts/04\_folium\_maps.py

python scripts/05\_eda\_plots.py

```



These scripts support dataset inspection, candidate generation, and visualization.



\## 6. Running Automated Tests



The repository includes tests for geometry and pipeline functionality.



Execute:



```bash

python -m pytest -q tests

```



A successful test run should finish without failed tests.



\## 7. Checking the Results



After running the scripts, inspect the generated results and visualizations.



Check whether:



\- Dataset files were loaded successfully.

\- Candidate road segments were generated.

\- GPS observations were processed correctly.

\- Output files and visualizations were created.

\- Automated tests passed.



\## 8. Model Evaluation



The completed PathSnap experiments compare five machine-learning classifiers.



The evaluation includes trip-based cross-validation, student-test performance, and road-continuity analysis.



Detailed experimental results are documented in `docs/MODEL\_EVALUATION.md`.



\## 9. Troubleshooting



\### Missing Python package



Install dependencies again:



```bash

python -m pip install -r requirements.txt

```



\### Dataset not found



Verify that the required dataset directories exist and contain the expected files.



\### Test failures



Read the pytest error output to identify the failing test and its cause. Do not assume that all tests pass without running them.



\### Git branch issues



Check your current branch and working-tree status:



```bash

git branch --show-current

git status

```



\## 10. Final Verification



Before submitting the project:



1\. Confirm that the repository contains the final implementation.

2\. Confirm that the README and documentation are accurate.

3\. Run the automated tests.

4\. Verify the notebook and technical report.

5\. Confirm that the required GitHub links are accessible.

6\. Ensure that the reserved evaluator-test data has not been used for training.



\## 11. Conclusion



Testing and reproducibility are essential for validating the PathSnap map-matching system.



This guide provides the basic steps for setting up the repository, running its available scripts, and checking the project before submission.

