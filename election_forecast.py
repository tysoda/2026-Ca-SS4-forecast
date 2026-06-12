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
COUNTY_POLYS = {
    "Alpine": [
        (-119.91,38.54),(-119.55,38.54),(-119.55,38.90),(-119.91,38.90),(-119.91,38.54)
    ],
    "Amador": [
        (-120.99,38.31),(-120.52,38.31),(-120.52,38.60),(-120.99,38.60),(-120.99,38.31)
    ],
    "Calaveras": [
        (-120.99,37.96),(-120.38,37.96),(-120.38,38.31),(-120.99,38.31),(-120.99,37.96)
    ],
    "El Dorado": [
        (-120.99,38.54),(-120.07,38.54),(-120.07,39.07),(-120.99,39.07),(-120.99,38.54)
    ],
    "Inyo": [
        (-118.36,35.79),(-117.02,35.79),(-117.02,37.52),(-118.36,37.52),(-118.36,35.79)
    ],
    "Madera": [
        (-120.38,36.96),(-119.08,36.96),(-119.08,37.62),(-120.38,37.62),(-120.38,36.96)
    ],
    "Mariposa": [
        (-120.38,37.34),(-119.50,37.34),(-119.50,37.96),(-120.38,37.96),(-120.38,37.34)
    ],
    "Merced": [
        (-121.25,36.98),(-120.05,36.98),(-120.05,37.48),(-121.25,37.48),(-121.25,36.98)
    ],
    "Mono": [
        (-119.55,37.46),(-118.36,37.46),(-118.36,38.51),(-119.55,38.51),(-119.55,37.46)
    ],
    "Nevada": [
        (-121.14,39.07),(-120.52,39.07),(-120.52,39.47),(-121.14,39.47),(-121.14,39.07)
    ],
    "Placer": [
        (-121.14,38.73),(-120.52,38.73),(-120.52,39.07),(-121.14,39.07),(-121.14,38.73)
    ],
    "Stanislaus": [
        (-121.35,37.18),(-120.05,37.18),(-120.05,37.76),(-121.35,37.76),(-121.35,37.18)
    ],
    "Tuolumne": [
        (-120.38,37.62),(-119.19,37.62),(-119.19,38.10),(-120.38,38.10),(-120.38,37.62)
    ],
}

COUNTY_CENTROIDS = {
    "Alpine":     (-119.73, 38.72),
    "Amador":     (-120.76, 38.46),
    "Calaveras":  (-120.69, 38.14),
    "El Dorado":  (-120.53, 38.81),
    "Inyo":       (-117.69, 36.66),
    "Madera":     (-119.73, 37.29),
    "Mariposa":   (-119.94, 37.65),
    "Merced":     (-120.65, 37.23),
    "Mono":       (-118.96, 37.99),
    "Nevada":     (-120.83, 39.27),
    "Placer":     (-120.83, 38.90),
    "Stanislaus": (-120.70, 37.47),
    "Tuolumne":   (-119.79, 37.86),
}

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
    st.markdown('<div class="section-label">County Win Probability by Vote Threshold · All Simulations</div>', unsafe_allow_html=True)
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
            "Win Probability (>45%)": lambda cn: float(np.mean(county_shares[cn] >= 0.45)),
            "Win Probability (>40%)": lambda cn: float(np.mean(county_shares[cn] >= 0.40)),
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

        # Build matplotlib map
        import matplotlib.cm as cm
        fig_map, ax_map = plt.subplots(figsize=(9, 8))
        fig_map.patch.set_facecolor("#f0f0ec")
        ax_map.set_facecolor("#d4e8f0")

        # Use RdBu: red = low (Rep), blue = high (Dem)
        cmap = plt.colormaps["RdBu"]

        for cn, poly_coords in COUNTY_POLYS.items():
            val    = county_values.get(cn, 0.5)
            t      = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color  = cmap(t)
            xs     = [p[0] for p in poly_coords]
            ys     = [p[1] for p in poly_coords]
            is_sel = (cn == selected_county)

            ax_map.fill(xs, ys, color=color, alpha=0.85,
                       linewidth=2.5 if is_sel else 0.8,
                       edgecolor="#ffffff" if not is_sel else "#FFD700")
            if is_sel:
                ax_map.fill(xs, ys, color="none", linewidth=3, edgecolor="#FFD700")

            # Label
            cx, cy = COUNTY_CENTROIDS.get(cn, (np.mean(xs), np.mean(ys)))
            short  = cn[:5] if len(cn) > 7 else cn

            is_pct = "Share" in selected_metric or "Average" in selected_metric or "Deviation" in selected_metric
            val_label = f"{val*100:.1f}%" if is_pct or "Probability" in selected_metric else f"{val:.3f}"

            ax_map.text(cx, cy + 0.04, short, ha="center", va="center",
                       fontsize=5.5, fontweight="bold", color="white",
                       path_effects=[pe.withStroke(linewidth=1.5, foreground="#333")])
            ax_map.text(cx, cy - 0.10, val_label, ha="center", va="center",
                       fontsize=5, color="white",
                       path_effects=[pe.withStroke(linewidth=1.2, foreground="#333")])

        ax_map.set_xlim(-121.8, -116.5)
        ax_map.set_ylim(35.5, 39.7)
        ax_map.set_aspect("equal")
        ax_map.axis("off")
        ax_map.set_title(f"SD4 — {selected_metric}", fontsize=9, pad=8, color="#333")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig_map.colorbar(sm, ax=ax_map, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        if "Probability" in selected_metric or "Share" in selected_metric or "Average" in selected_metric:
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
