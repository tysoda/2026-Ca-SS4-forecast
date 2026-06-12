import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import json, math

st.set_page_config(page_title="Election Forecast — SD4", page_icon="🗳️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em; }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; }
  .section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; color: #888;
    border-bottom: 1px solid #e0e0e0; padding-bottom: 0.4rem; margin-bottom: 1rem;
  }
  .stat-card { background: #f7f7f5; border-left: 3px solid #1a6b3c; padding: 0.85rem 1rem; border-radius: 2px; }
  .stat-card .label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #666; margin-bottom: 0.2rem; }
  .stat-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #111; line-height: 1; }
  .win-hero { background: linear-gradient(135deg, #0f3d24 0%, #1a6b3c 100%); color: white; padding: 1.5rem 1.75rem; border-radius: 4px; text-align: center; }
  .win-hero .wlabel { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.7; margin-bottom: 0.5rem; }
  .win-hero .wvalue { font-family: 'IBM Plex Mono', monospace; font-size: 3.2rem; font-weight: 600; line-height: 1; }
  .win-hero .wsub { font-size: 0.75rem; opacity: 0.6; margin-top: 0.5rem; }
  .styled-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .styled-table th { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.06em; text-transform: uppercase; color: #666; padding: 0.4rem 0.6rem; border-bottom: 2px solid #ddd; text-align: right; }
  .styled-table th:first-child { text-align: left; }
  .styled-table td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #f0f0f0; text-align: right; font-variant-numeric: tabular-nums; }
  .styled-table td:first-child { text-align: left; font-weight: 500; }
  .styled-table tr:hover td { background: #fafaf8; }
  .prob-high { color: #1a6b3c; font-weight: 600; }
  .prob-mid  { color: #d97706; font-weight: 600; }
  .prob-low  { color: #b91c1c; font-weight: 600; }
  .model-box { background: #f7f7f5; border: 1px solid #e8e8e4; border-radius: 4px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
  .model-box h4 { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #1a6b3c; margin: 0 0 0.6rem 0; }
  .model-box p { font-size: 0.85rem; color: #444; line-height: 1.6; margin: 0 0 0.5rem 0; }
  .model-box code { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; background: #eee; padding: 0.1rem 0.3rem; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── County geographic data ────────────────────────────────────────────────────
# Simplified polygon coordinates (lon, lat) for each county in SD4
# Approximated from official CA county boundaries
# ── SD4 boundary (from uploaded GeoJSON, simplified with RDP epsilon=0.005) ──
SD4_BOUNDARY = [(-116.47215, 36.44656), (-115.64837, 35.80922), (-115.73576, 35.8091), (-115.7359, 35.79363), (-117.91927, 35.7984), (-117.92302, 35.78669), (-118.00811, 35.78894), (-118.00774, 35.81709), (-117.99862, 35.82303), (-118.00621, 35.82904), (-118.00743, 35.85819), (-117.99671, 35.8695), (-117.98077, 35.86752), (-117.98908, 35.88131), (-117.98206, 35.8947), (-117.99033, 35.91405), (-117.9833, 35.92657), (-117.99196, 35.94378), (-118.0168, 35.95456), (-118.01458, 35.9729), (-118.00359, 35.98372), (-118.01209, 35.99831), (-118.03362, 36.00895), (-118.05173, 36.05955), (-118.05163, 36.08355), (-118.06733, 36.09345), (-118.07353, 36.14035), (-118.05973, 36.15085), (-118.05933, 36.17015), (-118.10563, 36.21344), (-118.10523, 36.23364), (-118.11938, 36.25557), (-118.1174, 36.27117), (-118.12761, 36.28035), (-118.12702, 36.30023), (-118.11183, 36.30834), (-118.11253, 36.32204), (-118.09763, 36.33114), (-118.10033, 36.34614), (-118.12423, 36.35194), (-118.13053, 36.37034), (-118.16293, 36.38944), (-118.14033, 36.40354), (-118.14123, 36.42094), (-118.15683, 36.43264), (-118.19363, 36.42654), (-118.21443, 36.43434), (-118.20963, 36.44214), (-118.21563, 36.45634), (-118.24993, 36.48244), (-118.23503, 36.49374), (-118.24143, 36.50004), (-118.23883, 36.52364), (-118.24943, 36.52404), (-118.25203, 36.54214), (-118.26534, 36.55154), (-118.29124, 36.55934), (-118.28904, 36.59074), (-118.27464, 36.59734), (-118.32134, 36.62744), (-118.31884, 36.63894), (-118.33824, 36.65544), (-118.33124, 36.66944), (-118.34754, 36.67234), (-118.36614, 36.69044), (-118.33554, 36.70444), (-118.33604, 36.71634), (-118.35094, 36.74074), (-118.36924, 36.75004), (-118.38024, 36.78224), (-118.37404, 36.80018), (-118.39374, 36.82967), (-118.36218, 36.84401), (-118.37019, 36.87169), (-118.36084, 36.88774), (-118.38844, 36.94554), (-118.40475, 36.95754), (-118.40445, 36.97204), (-118.41955, 36.98744), (-118.41225, 36.99834), (-118.42799, 37.01124), (-118.42277, 37.02581), (-118.44174, 37.04502), (-118.43498, 37.04974), (-118.43715, 37.05982), (-118.44897, 37.06914), (-118.46734, 37.06675), (-118.50312, 37.09523), (-118.52214, 37.09836), (-118.53094, 37.11119), (-118.5831, 37.1224), (-118.59267, 37.13815), (-118.65461, 37.14183), (-118.67266, 37.16735), (-118.66425, 37.17815), (-118.66647, 37.1895), (-118.68136, 37.20444), (-118.67563, 37.21415), (-118.68644, 37.22758), (-118.68376, 37.24437), (-118.66534, 37.26192), (-118.71603, 37.32821), (-118.74113, 37.31534), (-118.78675, 37.34339), (-118.78153, 37.35642), (-118.76816, 37.36118), (-118.79004, 37.39404), (-118.77977, 37.42168), (-118.76928, 37.42218), (-118.76019, 37.43364), (-118.7633, 37.45654), (-118.79711, 37.48872), (-118.85606, 37.4784), (-118.86077, 37.50154), (-118.90187, 37.52604), (-118.91707, 37.55034), (-118.92967, 37.54894), (-118.95267, 37.56584), (-118.97697, 37.55684), (-119.02208, 37.58584), (-119.3122, 37.35273), (-119.31224, 37.33971), (-119.32579, 37.33542), (-119.31576, 37.32478), (-119.33544, 37.31122), (-119.32695, 37.29191), (-119.33232, 37.27444), (-119.32219, 37.25342), (-119.3378, 37.21917), (-119.33009, 37.20743), (-119.34312, 37.18908), (-119.36058, 37.18054), (-119.36193, 37.16785), (-119.38886, 37.14922), (-119.41581, 37.16345), (-119.43214, 37.16258), (-119.43489, 37.14696), (-119.46265, 37.14422), (-119.47482, 37.10997), (-119.49133, 37.11972), (-119.48919, 37.13727), (-119.50606, 37.15035), (-119.51748, 37.14678), (-119.52499, 37.12824), (-119.55001, 37.14459), (-119.56085, 37.14291), (-119.56858, 37.11669), (-119.54877, 37.11676), (-119.5376, 37.10507), (-119.55901, 37.08806), (-119.56144, 37.06549), (-119.60492, 37.07102), (-119.62113, 37.02661), (-119.63538, 37.02155), (-119.62905, 37.03462), (-119.64947, 37.0435), (-119.65939, 37.03894), (-119.65862, 37.01334), (-119.69809, 37.00875), (-119.74051, 36.97022), (-119.74327, 36.95464), (-119.73378, 36.94645), (-119.75207, 36.93591), (-119.75496, 36.92246), (-119.77275, 36.9186), (-119.78932, 36.89671), (-119.78613, 36.8789), (-119.81876, 36.84807), (-119.8409, 36.8613), (-119.855, 36.85118), (-119.8849, 36.85855), (-119.91193, 36.84532), (-119.92911, 36.84791), (-119.94352, 36.83404), (-119.97055, 36.83287), (-119.98466, 36.84085), (-119.9924, 36.82895), (-120.01369, 36.82814), (-120.02779, 36.81451), (-120.07947, 36.82535), (-120.11491, 36.81434), (-120.11034, 36.87301), (-120.05597, 36.87312), (-120.05602, 36.91662), (-120.03758, 36.91665), (-120.03243, 36.93542), (-119.98754, 36.93786), (-120.01304, 36.96707), (-120.01056, 36.97432), (-119.99821, 36.97419), (-120.00092, 36.99451), (-120.02972, 36.98465), (-120.02415, 36.99163), (-120.02904, 37.0326), (-120.0017, 37.0328), (-120.00207, 37.07045), (-120.02934, 37.06675), (-120.02935, 37.0836), (-120.02207, 37.08358), (-120.03918, 37.10147), (-120.0347, 37.1116), (-120.05552, 37.12479), (-120.09456, 37.12859), (-120.12033, 37.1475), (-120.12154, 37.15911), (-120.10917, 37.1659), (-120.05207, 37.18311), (-120.09009, 37.22148), (-120.14384, 37.2392), (-120.17851, 37.26243), (-120.18772, 37.30108), (-120.28341, 37.42437), (-120.27498, 37.44364), (-120.28401, 37.4629), (-120.3116, 37.45561), (-120.37022, 37.42229), (-120.44951, 37.4011), (-120.45527, 37.41498), (-120.46889, 37.40599), (-120.4698, 37.41356), (-120.48257, 37.41839), (-120.48718, 37.40403), (-120.52135, 37.40392), (-120.52135, 37.41843), (-120.57598, 37.4184), (-120.57616, 37.4039), (-120.66786, 37.40391), (-120.66842, 37.36717), (-120.70969, 37.38231), (-120.70048, 37.38405), (-120.70479, 37.40044), (-120.72849, 37.40816), (-120.74263, 37.39934), (-120.74134, 37.39262), (-120.74607, 37.39925), (-120.76728, 37.38821), (-120.79175, 37.39229), (-120.82816, 37.38154), (-120.83575, 37.36921), (-120.86218, 37.35763), (-120.88109, 37.36275), (-120.89696, 37.35558), (-120.90773, 37.36364), (-120.9221, 37.35557), (-120.92409, 37.36969), (-120.94015, 37.3691), (-120.94686, 37.35897), (-120.95896, 37.35793), (-120.95651, 37.34917), (-121.22682, 37.13478), (-121.23712, 37.15721), (-121.26211, 37.15933), (-121.28112, 37.18361), (-121.29856, 37.16597), (-121.32842, 37.16595), (-121.36095, 37.18433), (-121.38428, 37.16621), (-121.38356, 37.15149), (-121.39903, 37.15014), (-121.41341, 37.17233), (-121.40881, 37.18251), (-121.42184, 37.22131), (-121.44176, 37.23113), (-121.45576, 37.24944), (-121.45908, 37.28232), (-121.44966, 37.29394), (-121.42347, 37.29529), (-121.40577, 37.31099), (-121.42366, 37.35884), (-121.40909, 37.38068), (-121.42406, 37.39364), (-121.45666, 37.39554), (-121.45636, 37.40674), (-121.47262, 37.42335), (-121.46187, 37.4388), (-121.46293, 37.45149), (-121.48679, 37.47566), (-121.2409, 37.66478), (-121.2302, 37.66298), (-121.21853, 37.67358), (-121.22348, 37.68361), (-121.20871, 37.68631), (-121.20246, 37.69599), (-121.18119, 37.6879), (-121.17987, 37.70415), (-121.16369, 37.70038), (-121.15511, 37.72033), (-121.12137, 37.72192), (-121.10764, 37.73229), (-121.11006, 37.74213), (-121.09538, 37.73323), (-121.0877, 37.74141), (-121.067, 37.73957), (-121.05675, 37.75052), (-121.044, 37.73871), (-121.02883, 37.74071), (-121.01773, 37.75528), (-121.00801, 37.7491), (-120.99345, 37.76095), (-120.95404, 37.73836), (-120.9216, 37.73765), (-120.91722, 37.75221), (-120.92341, 37.75803), (-120.92646, 38.07743), (-120.93886, 38.08833), (-121.0271, 38.30026), (-121.02744, 38.50815), (-121.11906, 38.71787), (-121.13317, 38.70538), (-121.14161, 38.71194), (-121.13452, 38.71204), (-121.11859, 38.76894), (-121.10176, 38.78798), (-121.10177, 38.81523), (-121.08497, 38.81602), (-121.08735, 38.83328), (-121.05842, 38.84713), (-121.06182, 38.85992), (-121.05328, 38.86835), (-121.06197, 38.88214), (-121.0443, 38.89034), (-121.05355, 38.89881), (-121.04042, 38.91566), (-121.00026, 38.91795), (-120.95778, 38.93911), (-120.93858, 38.93557), (-120.93622, 38.96393), (-120.90197, 38.95312), (-120.88794, 38.95974), (-120.85988, 38.95166), (-120.84991, 38.97655), (-120.8347, 38.9716), (-120.82474, 38.99275), (-120.8086, 39.001), (-120.78853, 38.99953), (-120.78599, 39.02772), (-120.80207, 39.02657), (-120.76942, 39.05876), (-120.79026, 39.07152), (-120.78273, 39.08071), (-120.7629, 39.07549), (-120.77072, 39.09034), (-120.76876, 39.10744), (-120.7351, 39.11911), (-120.71903, 39.11444), (-120.70166, 39.13593), (-120.68359, 39.13775), (-120.6924, 39.12193), (-120.68952, 39.11335), (-120.6703, 39.1116), (-120.67044, 39.09505), (-120.66519, 39.10168), (-120.63864, 39.10101), (-120.6171, 39.1162), (-120.63456, 39.11633), (-120.62161, 39.13458), (-120.61477, 39.18185), (-120.59859, 39.18724), (-120.56745, 39.19286), (-120.56882, 39.18328), (-120.5362, 39.18375), (-120.50563, 39.1735), (-120.50283, 39.15642), (-120.471, 39.17426), (-120.46049, 39.16582), (-120.43963, 39.17716), (-120.42561, 39.1999), (-120.43258, 39.21672), (-120.4439, 39.22268), (-120.43407, 39.24172), (-120.42064, 39.24348), (-120.41658, 39.25271), (-120.45391, 39.25865), (-120.44506, 39.29231), (-120.45733, 39.30554), (-120.44524, 39.30755), (-120.46445, 39.31614), (-120.46124, 39.32099), (-120.40793, 39.32905), (-120.40014, 39.33671), (-120.37389, 39.32702), (-120.33479, 39.34344), (-120.3059, 39.33587), (-120.27335, 39.35817), (-120.25472, 39.35868), (-120.25485, 39.37397), (-120.23654, 39.37415), (-120.22042, 39.36313), (-120.18169, 39.37259), (-120.16094, 39.36729), (-120.14936, 39.37473), (-120.11624, 39.36799), (-120.08587, 39.38733), (-120.0643, 39.37376), (-120.04098, 39.37043), (-120.02305, 39.37984), (-120.02116, 39.40014), (-120.00528, 39.40008), (-120.00103, 38.99958), (-118.92253, 38.24992), (-118.0522, 37.62494), (-117.24493, 37.03025), (-116.47215, 36.44656)]

# ── County centroids for label placement ─────────────────────────────────────
COUNTY_CENTROIDS = {
    "Alpine":     (-119.73, 38.60),
    "Amador":     (-120.65, 38.43),
    "Calaveras":  (-120.56, 38.14),
    "El Dorado":  (-120.52, 38.78),
    "Inyo":       (-117.69, 36.66),
    "Madera":     (-119.71, 37.22),
    "Mariposa":   (-119.91, 37.57),
    "Merced":     (-120.65, 37.23),
    "Mono":       (-118.96, 37.94),
    "Nevada":     (-120.83, 39.27),
    "Placer":     (-120.74, 38.90),
    "Stanislaus": (-120.70, 37.47),
    "Tuolumne":   (-119.94, 37.86),
}

SD4_COUNTIES = {
    "Alpine", "Amador", "Calaveras", "El Dorado", "Inyo",
    "Madera", "Mariposa", "Merced", "Mono", "Nevada",
    "Placer", "Stanislaus", "Tuolumne",
}

@st.cache_data(show_spinner="Loading county boundaries…", ttl=86400)
def load_county_geojson():
    """Fetch and cache CA county GeoJSON, filtering to SD4 counties."""
    import urllib.request, json as _json
    urls = [
        "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson",
        "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/california-counties.geojson",
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                data = _json.loads(r.read())
            # Filter to our 13 counties
            polys = {}
            for feat in data.get("features", []):
                props = feat.get("properties", {})
                # Try various name fields
                name = props.get("name") or props.get("NAME") or props.get("county") or ""
                name = name.replace(" County", "").strip()
                if name in SD4_COUNTIES:
                    geom = feat["geometry"]
                    # Normalise to list of rings (handle Polygon + MultiPolygon)
                    if geom["type"] == "Polygon":
                        rings = [geom["coordinates"][0]]
                    else:
                        rings = [poly[0] for poly in geom["coordinates"]]
                    # Pick largest ring by point count
                    ring = max(rings, key=len)
                    # Simplify with RDP
                    polys[name] = _rdp(ring, 0.002)
            if len(polys) >= 10:
                return polys
        except Exception:
            continue
    return {}  # fallback — map tab will show a warning

def _rdp(points, epsilon):
    """Ramer-Douglas-Peucker simplification."""
    import sys
    sys.setrecursionlimit(20000)
    def _rdp_inner(pts, eps):
        if len(pts) < 3:
            return list(pts)
        x1,y1 = float(pts[0][0]),float(pts[0][1])
        x2,y2 = float(pts[-1][0]),float(pts[-1][1])
        dx,dy = x2-x1, y2-y1
        max_d, max_i = 0, 0
        for i in range(1, len(pts)-1):
            x0,y0 = float(pts[i][0]),float(pts[i][1])
            if dx==0 and dy==0:
                d = ((x0-x1)**2+(y0-y1)**2)**0.5
            else:
                t = max(0,min(1,((x0-x1)*dx+(y0-y1)*dy)/(dx*dx+dy*dy)))
                d = ((x0-(x1+t*dx))**2+(y0-(y1+t*dy))**2)**0.5
            if d > max_d:
                max_d,max_i = d,i
        if max_d > eps:
            return _rdp_inner(pts[:max_i+1],eps)[:-1] + _rdp_inner(pts[max_i:],eps)
        return [pts[0], pts[-1]]
    return _rdp_inner(points, epsilon)


# ── Default model data ────────────────────────────────────────────────────────
DEFAULT_COUNTIES = {
    "Alpine":     {"registration": 944,    "turnout": 0.7074, "lean_avg": 0.0388, "lean_lin": 0.0995, "lean_sd": 0.0293, "hist_avg": 0.6221},
    "Amador":     {"registration": 27416,  "turnout": 0.7110, "lean_avg":-0.2465, "lean_lin":-0.2444, "lean_sd": 0.0143, "hist_avg": 0.3481},
    "Calaveras":  {"registration": 33312,  "turnout": 0.6702, "lean_avg":-0.2404, "lean_lin":-0.2538, "lean_sd": 0.0162, "hist_avg": 0.3581},
    "El Dorado":  {"registration": 142947, "turnout": 0.6630, "lean_avg":-0.1784, "lean_lin":-0.1531, "lean_sd": 0.0140, "hist_avg": 0.4079},
    "Inyo":       {"registration": 11037,  "turnout": 0.6890, "lean_avg":-0.1280, "lean_lin":-0.0973, "lean_sd": 0.0122, "hist_avg": 0.4570},
    "Madera":     {"registration": 34903,  "turnout": 0.5690, "lean_avg":-0.2700, "lean_lin":-0.2325, "lean_sd": 0.0182, "hist_avg": 0.3155},
    "Mariposa":   {"registration": 11862,  "turnout": 0.6941, "lean_avg":-0.2083, "lean_lin":-0.2038, "lean_sd": 0.0065, "hist_avg": 0.3772},
    "Merced":     {"registration": 12842,  "turnout": 0.4989, "lean_avg":-0.2673, "lean_lin":-0.2239, "lean_sd": 0.0231, "hist_avg": 0.3182},
    "Mono":       {"registration": 8286,   "turnout": 0.6654, "lean_avg":-0.0251, "lean_lin":-0.0033, "lean_sd": 0.0173, "hist_avg": 0.5628},
    "Nevada":     {"registration": 12680,  "turnout": 0.6959, "lean_avg": 0.1353, "lean_lin": 0.1741, "lean_sd": 0.0181, "hist_avg": 0.7208},
    "Placer":     {"registration": 8640,   "turnout": 0.6520, "lean_avg": 0.1224, "lean_lin": 0.1462, "lean_sd": 0.0113, "hist_avg": 0.7079},
    "Stanislaus": {"registration": 304987, "turnout": 0.5178, "lean_avg":-0.1553, "lean_lin":-0.1540, "lean_sd": 0.0123, "hist_avg": 0.4357},
    "Tuolumne":   {"registration": 36167,  "turnout": 0.6736, "lean_avg":-0.2092, "lean_lin":-0.2034, "lean_sd": 0.0140, "hist_avg": 0.3763},
}
DEFAULT_DISTRICT_TURNOUT_SD = 0.0823
DEFAULT_COUNTY_TURNOUT_SD   = 0.0773
DEFAULT_STATE_ENV_SD        = 0.0629
WIN_THRESHOLD               = 0.50
N_DEFAULT                   = 10_000

# ── Session state init ────────────────────────────────────────────────────────
if "county_df" not in st.session_state:
    rows = []
    for cn, d in DEFAULT_COUNTIES.items():
        rows.append({
            "County":       cn,
            "Registration": int(d["registration"]),
            "Turnout (%)":  round(d["turnout"]  * 100, 2),
            "Lean Avg (%)": round(d["lean_avg"] * 100, 2),
            "Lean Lin (%)": round(d["lean_lin"] * 100, 2),
            "Lean SD (%)":  round(d["lean_sd"]  * 100, 2),
            "Hist Avg (%)": round(d["hist_avg"] * 100, 2),
        })
    st.session_state["county_df"] = pd.DataFrame(rows)

for key, default in [
    ("district_turnout_sd_pct", round(DEFAULT_DISTRICT_TURNOUT_SD * 100, 2)),
    ("county_turnout_sd_pct",   round(DEFAULT_COUNTY_TURNOUT_SD   * 100, 2)),
    ("state_env_sd_pct",        round(DEFAULT_STATE_ENV_SD        * 100, 2)),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Live parameters from session state ───────────────────────────────────────
DISTRICT_TURNOUT_SD = st.session_state["district_turnout_sd_pct"] / 100
COUNTY_TURNOUT_SD   = st.session_state["county_turnout_sd_pct"]   / 100
STATE_ENV_SD        = st.session_state["state_env_sd_pct"]         / 100

COUNTIES = {}
for _, row in st.session_state["county_df"].iterrows():
    COUNTIES[row["County"]] = {
        "registration": int(row["Registration"]),
        "turnout":      float(row["Turnout (%)"]) / 100,
        "lean_avg":     float(row["Lean Avg (%)"]) / 100,
        "lean_lin":     float(row["Lean Lin (%)"]) / 100,
        "lean_sd":      float(row["Lean SD (%)"]) / 100,
        "hist_avg":     float(row["Hist Avg (%)"]) / 100,
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Simulation Parameters")
    n_sims       = st.number_input("Simulations", min_value=1000, max_value=100_000, value=N_DEFAULT, step=1000)
    forecast_env = st.number_input("Forecast State Environment (%)", value=59.80, step=0.1) / 100
    lean_method  = st.radio("County Lean Method", ["Average", "Linear"], horizontal=True)

    st.markdown("---")
    st.markdown("### 🔍 Conditional Filters")
    st.caption("County filters use the **vote threshold** below. District win probability always uses 50%.")
    county_vote_threshold = st.number_input("County Vote Threshold (%)", value=50.0, step=0.5) / 100

    county_filters = {}
    for county in COUNTIES:
        county_filters[county] = st.selectbox(
            county, ["Ignore", "Over threshold", "Under threshold"], key=f"filter_{county}"
        )

    env_lower = st.number_input("State Env Lower (%)", value=0.0,   step=1.0) / 100
    env_upper = st.number_input("State Env Upper (%)", value=100.0, step=1.0) / 100
    run = st.button("▶ Run Simulation", type="primary", width="stretch")

# ── Simulation ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running simulations…")
def run_simulation(n, forecast_env, lean_method_str, seed,
                   district_turnout_sd, county_turnout_sd, state_env_sd,
                   counties_frozen, turnout_adjustments_frozen):
    rng      = np.random.default_rng(seed)
    n        = int(n)
    counties = {name: dict(fields) for name, fields in counties_frozen}
    turnout_adj = dict(turnout_adjustments_frozen)

    env_error              = rng.normal(0, state_env_sd, n)
    sim_env                = forecast_env + env_error
    district_turnout_shift = rng.normal(0, district_turnout_sd, n)

    lean_key         = "lean_avg" if lean_method_str == "Average" else "lean_lin"
    county_shares    = {}
    county_votes     = {}
    county_dem_votes = {}
    county_turnouts  = {}

    for name, data in counties.items():
        adj = turnout_adj.get(name, 0.0)
        turnout_sim = np.clip(
            data["turnout"] + adj + district_turnout_shift + rng.normal(0, county_turnout_sd, n),
            0.01, 1.0
        )
        lean_error = rng.normal(0, data["lean_sd"], n)
        share      = np.clip(sim_env + data[lean_key] + lean_error, 0.0, 1.0)
        votes      = data["registration"] * turnout_sim
        dem_votes  = votes * share

        county_shares[name]    = share
        county_votes[name]     = votes
        county_dem_votes[name] = dem_votes
        county_turnouts[name]  = turnout_sim

    total_votes     = sum(county_votes.values())
    total_dem_votes = sum(county_dem_votes.values())
    district_share  = total_dem_votes / total_votes

    return {
        "district_share":   district_share,
        "sim_env":          sim_env,
        "env_error":        env_error,
        "county_shares":    county_shares,
        "county_votes":     county_votes,
        "county_dem_votes": county_dem_votes,
        "county_turnouts":  county_turnouts,
    }

def counties_to_frozen(d):
    return tuple((name, tuple(sorted(data.items()))) for name, data in d.items())

# ── Turnout adjustments (live sliders, not cached-only) ───────────────────────
if "turnout_adj" not in st.session_state:
    st.session_state["turnout_adj"] = {cn: 0.0 for cn in COUNTIES}

if "sim_results" not in st.session_state or run:
    seed = np.random.randint(0, 2**31)
    st.session_state["sim_results"] = run_simulation(
        n_sims, forecast_env, lean_method, seed,
        DISTRICT_TURNOUT_SD, COUNTY_TURNOUT_SD, STATE_ENV_SD,
        counties_to_frozen(COUNTIES),
        tuple(sorted(st.session_state["turnout_adj"].items()))
    )

res            = st.session_state["sim_results"]
district_share = res["district_share"]
sim_env        = res["sim_env"]
env_error      = res["env_error"]
county_shares  = res["county_shares"]
county_names   = list(COUNTIES.keys())

# ── Conditional filters ───────────────────────────────────────────────────────
mask = (sim_env >= env_lower) & (sim_env <= env_upper)
for county, filt in county_filters.items():
    if county not in county_shares: continue
    if filt == "Over threshold":
        mask &= county_shares[county] >= county_vote_threshold
    elif filt == "Under threshold":
        mask &= county_shares[county] < county_vote_threshold

filtered_share = district_share[mask]
n_filtered     = mask.sum()

# ── Helpers ───────────────────────────────────────────────────────────────────
def prob_class(p):
    if p >= 0.60: return "prob-high"
    if p >= 0.40: return "prob-mid"
    return "prob-low"

def fmt_pct(v, decimals=1):
    return f"{v*100:.{decimals}f}%"

def val_to_color(val, vmin, vmax, cmap_name="RdBu"):
    """Map a scalar value to an RGBA tuple using a matplotlib colormap."""
    import matplotlib.cm as cm
    cmap = plt.colormaps[cmap_name]
    if vmax == vmin:
        t = 0.5
    else:
        t = (val - vmin) / (vmax - vmin)
    return cmap(t)

# ══════════════════════════════════════════════════════════════════════════════
st.title("🗳️ SD4 Election Forecast")
st.caption(
    f"Monte Carlo · {int(n_sims):,} simulations · Lean: {lean_method} · "
    f"District win threshold: 50% (fixed) · County filter threshold: {fmt_pct(county_vote_threshold)}"
)

tab_dash, tab_turnout, tab_map, tab_hood = st.tabs([
    "📊 Dashboard", "🎚️ Turnout Explorer", "🗺️ District Map", "🔧 Under the Hood"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
    ds     = filtered_share if n_filtered > 0 else district_share
    wp     = float(np.mean(ds >= WIN_THRESHOLD))
    n_wins = int(np.sum(ds >= WIN_THRESHOLD))

    col_win, col_stats = st.columns([1, 3])
    with col_win:
        st.markdown(f"""
        <div class="win-hero">
          <div class="wlabel">Win Probability</div>
          <div class="wvalue">{fmt_pct(wp)}</div>
          <div class="wsub">{n_wins:,} of {n_filtered:,} simulations</div>
        </div>""", unsafe_allow_html=True)
    with col_stats:
        metrics = [
            ("Mean Share",      fmt_pct(np.mean(ds))),
            ("Median Share",    fmt_pct(np.median(ds))),
            ("Std Deviation",   fmt_pct(np.std(ds))),
            ("5th Percentile",  fmt_pct(np.percentile(ds, 5))),
            ("95th Percentile", fmt_pct(np.percentile(ds, 95))),
        ]
        cols = st.columns(5)
        for col, (label, val) in zip(cols, metrics):
            with col:
                st.markdown(f'<div class="stat-card"><div class="label">{label}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Result Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#f7f7f5"); ax.set_facecolor("#f7f7f5")
    ax.hist(ds * 100, bins=60, color="#1a6b3c", alpha=0.75, edgecolor="none")
    ax.axvline(WIN_THRESHOLD * 100, color="#b91c1c", linewidth=1.5, linestyle="--", label="Win threshold (50%)")
    ax.axvline(np.mean(ds) * 100, color="#0f3d24", linewidth=1.5, linestyle="-", label=f"Mean {fmt_pct(np.mean(ds))}")
    ax.set_xlabel("Dem Vote Share (%)", fontsize=9); ax.set_ylabel("Simulations", fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(labelsize=8); ax.legend(fontsize=8, framealpha=0)
    plt.tight_layout(); st.pyplot(fig, width="stretch"); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">County Probability of Reaching Threshold · All Simulations</div>', unsafe_allow_html=True)
    thresholds     = [0.40, 0.45, 0.50]
    county_headers = "".join(f"<th>{cn[:6]}</th>" for cn in county_names)
    rows_html      = ""
    for thr in thresholds:
        district_wp = np.mean(district_share >= WIN_THRESHOLD)
        row = f'<tr><td>&gt;{fmt_pct(thr, 0)}</td><td class="{prob_class(district_wp)}">{fmt_pct(district_wp)}</td>'
        for cn in county_names:
            cwp = np.mean(county_shares[cn] >= thr)
            row += f'<td class="{prob_class(cwp)}">{fmt_pct(cwp)}</td>'
        rows_html += row + '</tr>'
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>Threshold</th><th>District (50%)</th>{county_headers}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p style="font-size:0.72rem;color:#999;margin-top:0.4rem">
      District column reflects fixed 50% win threshold. County columns show P(county share &gt; row threshold).
    </p>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">County Statistics</div>', unsafe_allow_html=True)
    county_rows = ""
    for cn in county_names:
        data = COUNTIES[cn]; sim_sh = county_shares[cn]
        mean_s = np.mean(sim_sh); med_s = np.median(sim_sh)
        hist = data["hist_avg"]; resid = mean_s - hist
        county_rows += (f"<tr><td>{cn}</td><td>{fmt_pct(mean_s)}</td><td>{fmt_pct(med_s)}</td>"
                        f"<td>{fmt_pct(hist)}</td><td>{'▲' if resid >= 0 else '▼'} {fmt_pct(abs(resid))}</td>"
                        f"<td>{fmt_pct(data['lean_sd'])}</td></tr>")
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>County</th><th>Mean Share</th><th>Median Share</th><th>Hist. Average</th><th>Residual</th><th>Lean SD</th></tr></thead>
      <tbody>{county_rows}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Win Probability by State Environment</div>', unsafe_allow_html=True)
    bands = [("≤ 0",-np.inf,0.00),("+0 to +5",0.00,0.05),("+5 to +8",0.05,0.08),
             ("+8 to +10",0.08,0.10),("+10 to +12",0.10,0.12),("> +12",0.12,np.inf)]
    env_rows = ""
    for label, lo, hi in bands:
        bm = (env_error >= lo) & (env_error < hi); nb = bm.sum()
        nw = int(np.sum(district_share[bm] >= WIN_THRESHOLD)) if nb > 0 else 0
        wb = nw / nb if nb > 0 else 0.0
        env_rows += f"<tr><td>{label}</td><td>{nw:,}</td><td>{nb:,}</td><td class='{prob_class(wb)}'>{fmt_pct(wb)}</td></tr>"
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>Env. vs Forecast</th><th>Wins</th><th>Simulations</th><th>Win Probability</th></tr></thead>
      <tbody>{env_rows}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Path to Victory</div>', unsafe_allow_html=True)
    win_mask = district_share >= WIN_THRESHOLD; n_wins_total = win_mask.sum()
    ptv_rows = ""
    for cn in county_names:
        county_won = county_shares[cn] >= WIN_THRESHOLD
        p_cw = (county_won & win_mask).sum() / n_wins_total if n_wins_total > 0 else 0.0
        corr_env = np.corrcoef(county_shares[cn], sim_env)[0, 1]
        ptv_rows += f"<tr><td>{cn}</td><td>{fmt_pct(p_cw)}</td><td>{corr_env:.4f}</td></tr>"
    corr_ds_env_err = np.corrcoef(district_share, env_error)[0, 1]
    corr_ds_env     = np.corrcoef(district_share, sim_env)[0, 1]
    col_ptv, col_corr = st.columns([2, 1])
    with col_ptv:
        st.markdown(f"""
        <table class="styled-table">
          <thead><tr><th>County</th><th>P(County Won | District Won)</th><th>Corr: County Share vs Env</th></tr></thead>
          <tbody>{ptv_rows}</tbody>
        </table>""", unsafe_allow_html=True)
    with col_corr:
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:0.75rem">
          <div class="label">District Share vs Env Error</div>
          <div class="value" style="font-size:1.3rem">{corr_ds_env_err:.4f}</div>
        </div>
        <div class="stat-card">
          <div class="label">District Share vs Environment</div>
          <div class="value" style="font-size:1.3rem">{corr_ds_env:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Turnout Analysis</div>', unsafe_allow_html=True)
    fig2, axes = plt.subplots(3, 5, figsize=(14, 6))
    fig2.patch.set_facecolor("#f7f7f5"); axes_flat = axes.flatten()
    for i, cn in enumerate(county_names):
        sim_t = res["county_votes"][cn] / COUNTIES[cn]["registration"]
        ax = axes_flat[i]; ax.set_facecolor("#f7f7f5")
        ax.hist(sim_t * 100, bins=30, color="#1a6b3c", alpha=0.7, edgecolor="none")
        ax.axvline(COUNTIES[cn]["turnout"] * 100, color="#b91c1c", linewidth=1.2, linestyle="--")
        ax.set_title(cn, fontsize=7, fontweight="600"); ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.spines[["top","right","left"]].set_visible(False)
    for j in range(len(county_names), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Simulated Turnout by County  (red = forecast)", fontsize=8, color="#555")
    plt.tight_layout(); st.pyplot(fig2, width="stretch"); plt.close()

    st.markdown("---")
    st.caption(
        f"Filtered simulations: {n_filtered:,} of {int(n_sims):,} · "
        f"State env range: {fmt_pct(env_lower)} – {fmt_pct(env_upper)} · "
        f"County filters active: {sum(1 for v in county_filters.values() if v != 'Ignore')}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TURNOUT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_turnout:
    st.markdown("## Turnout Explorer")
    st.markdown(
        "Adjust turnout for individual counties relative to their forecast values. "
        "Positive values increase turnout, negative values decrease it. "
        "Hit **▶ Run Simulation** in the sidebar to apply adjustments. "
        "The impact panel below updates live to show what the adjusted baseline would mean "
        "for expected vote totals."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">County Turnout Adjustments (vs forecast)</div>', unsafe_allow_html=True)

    # Sliders in a 3-column grid
    adj_cols = st.columns(3)
    new_adj = {}
    for i, cn in enumerate(county_names):
        with adj_cols[i % 3]:
            base = COUNTIES[cn]["turnout"] * 100
            current_adj = st.session_state["turnout_adj"].get(cn, 0.0)
            adj_pct = st.slider(
                f"{cn}",
                min_value=-10.0, max_value=10.0,
                value=float(current_adj * 100),
                step=0.5,
                format="%.1f%%",
                help=f"Forecast: {base:.1f}%  →  Adjusted: {base + current_adj*100:.1f}%",
                key=f"turnout_slider_{cn}"
            )
            new_adj[cn] = adj_pct / 100

    st.session_state["turnout_adj"] = new_adj

    if st.button("↩ Reset all turnout adjustments"):
        st.session_state["turnout_adj"] = {cn: 0.0 for cn in COUNTIES}
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Projected Impact (based on current simulation)</div>', unsafe_allow_html=True)
    st.caption("Shows how adjusted turnout would change expected vote totals vs. the current simulation's mean. Re-run the simulation to see adjusted win probability.")

    # Compute simple deterministic impact (registration × adjusted_turnout × mean_share)
    impact_rows = ""
    total_base_votes = 0
    total_adj_votes  = 0
    total_base_dem   = 0
    total_adj_dem    = 0

    for cn in county_names:
        data     = COUNTIES[cn]
        adj      = new_adj.get(cn, 0.0)
        mean_sh  = float(np.mean(county_shares[cn]))
        base_t   = data["turnout"]
        adj_t    = np.clip(base_t + adj, 0.01, 1.0)
        reg      = data["registration"]

        base_votes = reg * base_t
        adj_votes  = reg * adj_t
        base_dem   = base_votes * mean_sh
        adj_dem    = adj_votes  * mean_sh
        delta_dem  = adj_dem - base_dem
        delta_str  = f"{'▲' if delta_dem >= 0 else '▼'} {abs(delta_dem):,.0f}"

        total_base_votes += base_votes; total_adj_votes  += adj_votes
        total_base_dem   += base_dem;   total_adj_dem    += adj_dem

        impact_rows += (
            f"<tr><td>{cn}</td>"
            f"<td>{fmt_pct(base_t)}</td>"
            f"<td>{fmt_pct(adj_t)}</td>"
            f"<td>{base_votes:,.0f}</td>"
            f"<td>{adj_votes:,.0f}</td>"
            f"<td>{fmt_pct(mean_sh)}</td>"
            f"<td>{delta_str}</td></tr>"
        )

    total_base_share = total_base_dem / total_base_votes if total_base_votes > 0 else 0
    total_adj_share  = total_adj_dem  / total_adj_votes  if total_adj_votes  > 0 else 0
    delta_share      = total_adj_share - total_base_share

    st.markdown(f"""
    <table class="styled-table">
      <thead><tr>
        <th>County</th><th>Base Turnout</th><th>Adj. Turnout</th>
        <th>Base Votes</th><th>Adj. Votes</th>
        <th>Mean Dem Share</th><th>Δ Dem Votes</th>
      </tr></thead>
      <tbody>{impact_rows}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    imp_c1, imp_c2, imp_c3 = st.columns(3)
    with imp_c1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="label">Base District Share</div>
          <div class="value">{fmt_pct(total_base_share)}</div>
        </div>""", unsafe_allow_html=True)
    with imp_c2:
        st.markdown(f"""
        <div class="stat-card">
          <div class="label">Adjusted District Share</div>
          <div class="value">{fmt_pct(total_adj_share)}</div>
        </div>""", unsafe_allow_html=True)
    with imp_c3:
        arrow = "▲" if delta_share >= 0 else "▼"
        st.markdown(f"""
        <div class="stat-card">
          <div class="label">Share Change</div>
          <div class="value">{arrow} {fmt_pct(abs(delta_share))}</div>
        </div>""", unsafe_allow_html=True)

    st.info("💡 Hit **▶ Run Simulation** in the sidebar to run the full Monte Carlo with these turnout adjustments and see the updated win probability.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRICT MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("## District Map — SD4")

    map_col, info_col = st.columns([2, 1])

    with map_col:
        metric_options = {
            "Mean Dem Vote Share":    lambda cn: float(np.mean(county_shares[cn])),
            "Win Probability (>50%)": lambda cn: float(np.mean(county_shares[cn] >= WIN_THRESHOLD)),
            "Vote Share (>45%)": lambda cn: float(np.mean(county_shares[cn] >= 0.45)),
            "Vote Share (>40%)": lambda cn: float(np.mean(county_shares[cn] >= 0.40)),
            "Median Dem Vote Share":  lambda cn: float(np.median(county_shares[cn])),
            "Std Deviation":          lambda cn: float(np.std(county_shares[cn])),
            "Historical Average":     lambda cn: COUNTIES[cn]["hist_avg"],
        }
        selected_metric = st.selectbox("Choropleth metric", list(metric_options.keys()))
        metric_fn = metric_options[selected_metric]

        county_values = {cn: metric_fn(cn) for cn in county_names}
        vmin = min(county_values.values())
        vmax = max(county_values.values())

        selected_county = st.session_state.get("selected_county", None)

        # Load real county boundaries (fetched and cached at runtime)
        county_polys = load_county_geojson()

        cmap = plt.colormaps["bwr_r"]
        fig_map, ax_map = plt.subplots(figsize=(9, 8))
        fig_map.patch.set_facecolor("#f0f0ec")
        ax_map.set_facecolor("#C8DFF0")

        # Draw SD4 district boundary as backdrop
        sd4_xs = [p[0] for p in SD4_BOUNDARY]
        sd4_ys = [p[1] for p in SD4_BOUNDARY]
        ax_map.fill(sd4_xs, sd4_ys, color="#e8e4dc", alpha=1.0, zorder=0)
        ax_map.plot(sd4_xs, sd4_ys, color="#222", linewidth=1.5, zorder=3)

        if not county_polys:
            ax_map.text(0.5, 0.5, "County boundaries unavailable.\nCheck internet connection.",
                       transform=ax_map.transAxes, ha="center", va="center", fontsize=10, color="#666")
        else:
            texts = []
            for cn in county_names:
                if cn not in county_polys:
                    continue
                ring   = county_polys[cn]
                val    = county_values.get(cn, 0.5)
                t      = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color  = cmap(t)
                xs     = [float(p[0]) for p in ring]
                ys     = [float(p[1]) for p in ring]
                is_sel = (cn == selected_county)
                ax_map.fill(xs, ys, color=color, alpha=0.88, zorder=1,
                            linewidth=2.5 if is_sel else 0.6,
                            edgecolor="#FFD700" if is_sel else "#ffffff")
                if is_sel:
                    ax_map.plot(xs, ys, color="#FFD700", linewidth=2.5, zorder=4)
                # Label at centroid
                cx, cy = COUNTY_CENTROIDS.get(cn, (np.mean(xs), np.mean(ys)))
                val_label = f"{val*100:.1f}%"
                texts.append(
                    ax_map.text(cx, cy, f"{cn}\n{val_label}", ha="center", va="center",
                                fontsize=5.5, fontweight="bold", color="white", zorder=5,
                                path_effects=[pe.withStroke(linewidth=1.8, foreground="#222")],
                                multialignment="center")
                )
            from adjustText import adjust_text
            adjust_text(
                texts,
                ax=ax_map,
                expand=(1.3, 1.5),
                force_text=(0.5, 0.8),
                force_static=(0.3, 0.5),
                arrowprops=dict(arrowstyle="-", color="#444", lw=0.6),
            )
        ax_map.set_xlim(-122.0, -115.4)
        ax_map.set_ylim(35.6, 39.6)
        ax_map.set_aspect("equal")
        ax_map.axis("off")
        ax_map.set_title(f"CA Senate District 4 — {selected_metric}", fontsize=9, pad=8, color="#333")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig_map.colorbar(sm, ax=ax_map, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        if any(k in selected_metric for k in ("Probability", "Share", "Average")):
            cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

        plt.tight_layout()
        st.pyplot(fig_map, width="stretch")
        plt.close()

    with info_col:
        st.markdown("### County Detail")
        selected_county = st.selectbox(
            "Select county", ["— select —"] + county_names,
            key="county_selector"
        )
        st.session_state["selected_county"] = selected_county if selected_county != "— select —" else None

        if selected_county and selected_county != "— select —":
            cn   = selected_county
            data = COUNTIES[cn]
            shr  = county_shares[cn]
            win_mask_c = district_share >= WIN_THRESHOLD

            mean_s  = float(np.mean(shr))
            med_s   = float(np.median(shr))
            sd_s    = float(np.std(shr))
            p5_s    = float(np.percentile(shr, 5))
            p95_s   = float(np.percentile(shr, 95))
            wp40    = float(np.mean(shr >= 0.40))
            wp45    = float(np.mean(shr >= 0.45))
            wp50    = float(np.mean(shr >= 0.50))
            hist    = data["hist_avg"]
            resid   = mean_s - hist
            county_won = shr >= WIN_THRESHOLD
            p_cw    = float((county_won & win_mask_c).sum() / win_mask_c.sum()) if win_mask_c.sum() > 0 else 0.0
            corr_e  = float(np.corrcoef(shr, sim_env)[0, 1])
            adj_t   = st.session_state["turnout_adj"].get(cn, 0.0)

            st.markdown(f"""
            <table class="styled-table" style="font-size:0.78rem;">
              <tbody>
                <tr><td><b>Registration</b></td><td>{data['registration']:,}</td></tr>
                <tr><td><b>Forecast turnout</b></td><td>{fmt_pct(data['turnout'])}</td></tr>
                <tr><td><b>Turnout adjustment</b></td><td>{'+' if adj_t >= 0 else ''}{fmt_pct(adj_t)}</td></tr>
                <tr><td colspan="2" style="padding-top:0.6rem;font-weight:600;color:#1a6b3c">Simulated Vote Share</td></tr>
                <tr><td>Mean</td><td>{fmt_pct(mean_s)}</td></tr>
                <tr><td>Median</td><td>{fmt_pct(med_s)}</td></tr>
                <tr><td>Std Dev</td><td>{fmt_pct(sd_s)}</td></tr>
                <tr><td>5th pct</td><td>{fmt_pct(p5_s)}</td></tr>
                <tr><td>95th pct</td><td>{fmt_pct(p95_s)}</td></tr>
                <tr><td colspan="2" style="padding-top:0.6rem;font-weight:600;color:#1a6b3c">Win Probabilities</td></tr>
                <tr><td>P(share &gt; 40%)</td><td>{fmt_pct(wp40)}</td></tr>
                <tr><td>P(share &gt; 45%)</td><td>{fmt_pct(wp45)}</td></tr>
                <tr><td>P(share &gt; 50%)</td><td>{fmt_pct(wp50)}</td></tr>
                <tr><td colspan="2" style="padding-top:0.6rem;font-weight:600;color:#1a6b3c">Historical</td></tr>
                <tr><td>Historical avg</td><td>{fmt_pct(hist)}</td></tr>
                <tr><td>Residual</td><td>{'▲' if resid >= 0 else '▼'} {fmt_pct(abs(resid))}</td></tr>
                <tr><td>Lean SD</td><td>{fmt_pct(data['lean_sd'])}</td></tr>
                <tr><td colspan="2" style="padding-top:0.6rem;font-weight:600;color:#1a6b3c">Path to Victory</td></tr>
                <tr><td>P(county won | district won)</td><td>{fmt_pct(p_cw)}</td></tr>
                <tr><td>Corr w/ state env</td><td>{corr_e:.4f}</td></tr>
              </tbody>
            </table>""", unsafe_allow_html=True)

            # Mini distribution
            st.markdown("<br>", unsafe_allow_html=True)
            fig_mini, ax_mini = plt.subplots(figsize=(4, 2))
            fig_mini.patch.set_facecolor("#f7f7f5"); ax_mini.set_facecolor("#f7f7f5")
            ax_mini.hist(shr * 100, bins=40, color="#1a6b3c", alpha=0.7, edgecolor="none")
            ax_mini.axvline(50, color="#b91c1c", linewidth=1.2, linestyle="--")
            ax_mini.set_title(f"{cn} vote share distribution", fontsize=7)
            ax_mini.xaxis.set_major_formatter(mtick.PercentFormatter())
            ax_mini.spines[["top","right","left"]].set_visible(False)
            ax_mini.tick_params(labelsize=6)
            plt.tight_layout()
            st.pyplot(fig_mini, width="stretch")
            plt.close()
        else:
            st.info("Select a county from the dropdown to see detailed statistics.")
            st.markdown(f"""
            <br>
            <div class="stat-card">
              <div class="label">District Win Probability</div>
              <div class="value">{fmt_pct(float(np.mean(district_share >= WIN_THRESHOLD)))}</div>
              <div class="sub">across all {int(n_sims):,} simulations</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — UNDER THE HOOD
# ══════════════════════════════════════════════════════════════════════════════
with tab_hood:
    st.markdown("## How the model works")
    st.markdown(
        f"County-level Monte Carlo simulation. Each of the **{int(n_sims):,} simulations** "
        "independently draws random values for the state environment, county turnouts, and county "
        "vote shares, then rolls them up to a single district result."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Global Standard Deviations</div>', unsafe_allow_html=True)
    st.caption("Edit values below, then hit **▶ Run Simulation** in the sidebar to apply.")

    sd_col1, sd_col2, sd_col3, _ = st.columns([1, 1, 1, 2])
    with sd_col1:
        v = st.number_input("District Turnout SD (%)", min_value=0.0, max_value=50.0, step=0.01,
                            value=float(st.session_state["district_turnout_sd_pct"]), format="%.2f", key="input_dist_sd")
        st.session_state["district_turnout_sd_pct"] = v
    with sd_col2:
        v = st.number_input("County Turnout SD (%)", min_value=0.0, max_value=50.0, step=0.01,
                            value=float(st.session_state["county_turnout_sd_pct"]), format="%.2f", key="input_county_sd")
        st.session_state["county_turnout_sd_pct"] = v
    with sd_col3:
        v = st.number_input("State Env SD (%)", min_value=0.0, max_value=50.0, step=0.01,
                            value=float(st.session_state["state_env_sd_pct"]), format="%.2f", key="input_env_sd")
        st.session_state["state_env_sd_pct"] = v

    if st.button("↩ Reset SDs to defaults", key="reset_sds"):
        st.session_state["district_turnout_sd_pct"] = round(DEFAULT_DISTRICT_TURNOUT_SD * 100, 2)
        st.session_state["county_turnout_sd_pct"]   = round(DEFAULT_COUNTY_TURNOUT_SD   * 100, 2)
        st.session_state["state_env_sd_pct"]        = round(DEFAULT_STATE_ENV_SD        * 100, 2)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">County Parameters</div>', unsafe_allow_html=True)
    st.caption("Edit values and click **Apply Changes** to update. Hit ▶ Run Simulation to re-run the model.")

    def apply_county_edits():
        """Called immediately when data_editor detects a change."""
        editor_state = st.session_state.get("county_editor", {})
        if not editor_state:
            return
        df = st.session_state["county_df"].copy()
        for idx_str, changes in editor_state.get("edited_rows", {}).items():
            idx = int(idx_str)
            for col, val in changes.items():
                df.at[idx, col] = val
        for row in editor_state.get("added_rows", []):
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        del_indices = editor_state.get("deleted_rows", [])
        if del_indices:
            df = df.drop(index=del_indices).reset_index(drop=True)
        st.session_state["county_df"] = df

    st.data_editor(
        st.session_state["county_df"],
        width="stretch", hide_index=True, disabled=["County"],
        column_config={
            "County":       st.column_config.TextColumn("County", width="medium"),
            "Registration": st.column_config.NumberColumn("Registration", min_value=0, step=1, format="%d"),
            "Turnout (%)":  st.column_config.NumberColumn("Turnout (%)",  min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "Lean Avg (%)": st.column_config.NumberColumn("Lean Avg (%)", min_value=-100.0, max_value=100.0, step=0.01, format="%.2f"),
            "Lean Lin (%)": st.column_config.NumberColumn("Lean Lin (%)", min_value=-100.0, max_value=100.0, step=0.01, format="%.2f"),
            "Lean SD (%)":  st.column_config.NumberColumn("Lean SD (%)",  min_value=0.0, max_value=50.0, step=0.01, format="%.2f"),
            "Hist Avg (%)": st.column_config.NumberColumn("Hist Avg (%)", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        },
        key="county_editor",
        on_change=apply_county_edits,
    )

    if st.button("↩ Reset county data to defaults", key="reset_counties"):
        rows = []
        for cn, d in DEFAULT_COUNTIES.items():
            rows.append({"County": cn, "Registration": int(d["registration"]),
                         "Turnout (%)": round(d["turnout"]*100,2), "Lean Avg (%)": round(d["lean_avg"]*100,2),
                         "Lean Lin (%)": round(d["lean_lin"]*100,2), "Lean SD (%)": round(d["lean_sd"]*100,2),
                         "Hist Avg (%)": round(d["hist_avg"]*100,2)})
        st.session_state["county_df"] = pd.DataFrame(rows)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Lean Method Comparison</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    fig3.patch.set_facecolor("#f7f7f5"); ax3.set_facecolor("#f7f7f5")
    x = np.arange(len(county_names)); w = 0.35
    ax3.bar(x - w/2, [COUNTIES[cn]["lean_avg"]*100 for cn in county_names], w, label="Average lean", color="#1a6b3c", alpha=0.8)
    ax3.bar(x + w/2, [COUNTIES[cn]["lean_lin"]*100 for cn in county_names], w, label="Linear lean",  color="#0f3d24", alpha=0.5)
    ax3.axhline(0, color="#ccc", linewidth=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels(county_names, rotation=40, ha="right", fontsize=7)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.set_ylabel("Lean vs State Env (%)", fontsize=8)
    ax3.legend(fontsize=8, framealpha=0); ax3.spines[["top","right","left"]].set_visible(False)
    ax3.tick_params(labelsize=7); plt.tight_layout()
    st.pyplot(fig3, width="stretch"); plt.close()
