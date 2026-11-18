"""Load the candidate tables built by scripts/03_build_candidates.py, with the team decision applied:
suspect-reference TRAIN trips (currently 33 and 54) are dropped from training/CV. The rule is data-driven
(trip_reference_quality.csv), no trip IDs are hardcoded. student_test is never filtered."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
QUALITY_CSV = os.path.join(ROOT, "results", "trip_reference_quality.csv")


def suspect_trips():
    q = pd.read_csv(QUALITY_CSV)
    return sorted(q.loc[(q.partition == "train") & (q.ref_quality == "suspect"), "trip_id"].tolist())


def load_candidates(partition, drop_suspect=True):
    df = pd.read_parquet(os.path.join(PROCESSED, f"candidates_{partition}.parquet"))
    if partition == "train" and drop_suspect:
        df = df[~df.trip_id.isin(suspect_trips())]
    return df.reset_index(drop=True)


def load_points(partition, drop_suspect=True):
    df = pd.read_parquet(os.path.join(PROCESSED, f"points_{partition}.parquet"))
    if partition == "train" and drop_suspect:
        df = df[~df.trip_id.isin(suspect_trips())]
    return df.reset_index(drop=True)
