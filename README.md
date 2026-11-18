# PathSnap

Map-matching of noisy GPS to a road network (UE24CS352A hackathon). Part A (data, EDA, candidates, labels, maps) is in place; other parts are added by teammates.

```bash
pip install -r requirements.txt
# put train/ and student_test/ under data/raw/ (git-ignored)
cd scripts
python 01_inventory.py && python 02_radius_sweep.py && python 03_build_candidates.py 80 15
python 04_folium_maps.py && python 05_eda_plots.py
python -m pytest -q ../tests
```
See `docs/PERSON_A_GUIDE.md` for data findings and design decisions.
