import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)
RESULTS = os.path.join(ROOT, "results")
PROCESSED = os.path.join(ROOT, "data", "processed")
SEED = 42
