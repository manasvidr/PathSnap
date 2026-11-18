"""Road graph for the HMM (Person C). Nodes = node ids, edge weight = arc length in metres (min over duplicate arcs).
Transition distance from arc a to arc b = shortest_path(arcs.v[a] -> arcs.u[b]); same arc = 0."""
import networkx as nx
from .geo import haversine


def build_digraph(trip):
    nd, ar = trip["nodes"], trip["arcs"]
    u, v = ar.u.values, ar.v.values
    length = haversine(nd.lon.values[u], nd.lat.values[u], nd.lon.values[v], nd.lat.values[v])
    G = nx.DiGraph()
    for a, b, w in zip(u, v, length):
        if a == b:
            continue                      # self-loops carry no movement
        if not G.has_edge(a, b) or w < G[a][b]["weight"]:
            G.add_edge(a, b, weight=float(w))
    return G
