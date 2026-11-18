"""Folium trip maps. Layers: raw GPS (blue), road network near the track (grey), reference route (green),
predicted route (red, optional). Network is clipped to arcs within `net_radius_m` of the track (full networks have up to 300k arcs)."""
import numpy as np
import folium
import shapely
from shapely import STRtree
from ..geo import LocalProj, arc_xy

LEGEND = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 10px;
border:1px solid #999;border-radius:4px;font:12px sans-serif">
<b>Trip {tid}</b><br>
<span style="color:#1f77b4">&#9679;</span> Raw GPS<br>
<span style="color:#888">&#9473;</span> Road network<br>
<span style="color:#2ca02c">&#9473;</span> Reference route<br>
{pred}</div>"""


def _arc_lines(trip, arc_ids):
    nd, ar = trip["nodes"], trip["arcs"]
    u, v = ar.u.values[arc_ids], ar.v.values[arc_ids]
    return [[(nd.lat.values[a], nd.lon.values[a]), (nd.lat.values[b], nd.lon.values[b])] for a, b in zip(u, v)]


def make_trip_map(trip, predicted_arcs=None, out_html=None, net_radius_m=120, max_net_arcs=15000, max_gps=4000):
    tr = trip["track"]
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    # network near the track
    proj = LocalProj.for_trip(trip)
    ax, ay, bx, by = arc_xy(trip, proj)
    segs = shapely.linestrings(np.stack([np.stack([ax, ay], 1), np.stack([bx, by], 1)], 1))
    px, py = proj.xy(tr.lon.values, tr.lat.values)
    step = max(1, len(px) // 3000)
    _, near = STRtree(segs).query(shapely.points(px[::step], py[::step]), predicate="dwithin", distance=net_radius_m)
    near = np.unique(near)
    if len(near) > max_net_arcs:
        near = near[np.linspace(0, len(near) - 1, max_net_arcs).astype(int)]
    g = folium.FeatureGroup("Road network", show=True)
    folium.PolyLine(_arc_lines(trip, near), color="#888", weight=1, opacity=0.7).add_to(g); g.add_to(m)
    # reference route
    if trip.get("route") is not None:
        g = folium.FeatureGroup("Reference route", show=True)
        folium.PolyLine(_arc_lines(trip, trip["route"].arc_id.values), color="#2ca02c", weight=4, opacity=0.8).add_to(g); g.add_to(m)
    # predicted route
    if predicted_arcs is not None:
        g = folium.FeatureGroup("Predicted route", show=True)
        folium.PolyLine(_arc_lines(trip, np.asarray(predicted_arcs)), color="#d62728", weight=3, opacity=0.9).add_to(g); g.add_to(m)
    # raw GPS
    g = folium.FeatureGroup("Raw GPS", show=True)
    s = max(1, len(tr) // max_gps)
    for i in range(0, len(tr), s):
        folium.CircleMarker((tr.lat.iloc[i], tr.lon.iloc[i]), radius=2, color="#1f77b4", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"pt {i}  t={tr.t.iloc[i]:.0f}s").add_to(g)
    g.add_to(m)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    pred = '<span style="color:#d62728">&#9473;</span> Predicted route' if predicted_arcs is not None else ""
    m.get_root().html.add_child(folium.Element(LEGEND.format(tid=trip["trip_id"], pred=pred)))
    if out_html:
        m.save(out_html)
    return m
