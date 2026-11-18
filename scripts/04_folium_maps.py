"""Example maps: a clean trip, the suspect-reference trips (54, 33) and a gap-heavy trip -> results/maps/*.html"""
import os, sys
from _common import RESULTS
from src.io_utils import list_trips, load_trip
from src.viz.folium_maps import make_trip_map

picks = [("train", "00000001"), ("train", "00000054"), ("train", "00000033"), ("student_test", "00000076")]
if len(sys.argv) > 1: picks = [("train" if f"{int(a):08d}" in list_trips("train") else "student_test", f"{int(a):08d}") for a in sys.argv[1:]]
for part, name in picks:
    make_trip_map(load_trip(part, name), basemap=os.environ.get("PATHSNAP_BASEMAP", "esri") or None, out_html=f"{RESULTS}/maps/trip_{name}.html"); print("saved", name)
