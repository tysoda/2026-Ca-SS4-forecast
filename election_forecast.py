import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Election Forecast", page_icon="🗳️", layout="wide")

# ── Styling ───────────────────────────────────────────────────────────────────
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
  .stat-card .sub { font-size: 0.72rem; color: #888; margin-top: 0.25rem; }
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
  .model-box h4 { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #1a6b3c; margin: 0 0 0.6rem 0; letter-spacing: 0.04em; }
  .model-box p { font-size: 0.85rem; color: #444; line-height: 1.6; margin: 0 0 0.5rem 0; }
  .model-box code { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; background: #eee; padding: 0.1rem 0.3rem; border-radius: 2px; }
  .param-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-top: 0.5rem; }
  .param-item { background: white; border: 1px solid #e0e0e0; border-radius: 3px; padding: 0.5rem 0.75rem; }
  .param-item .pname { font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; }
  .param-item .pval  { font-family: 'IBM Plex Mono', monospace; font-size: 1rem; font-weight: 600; color: #111; }
</style>
""", unsafe_allow_html=True)

# ── Default county data ───────────────────────────────────────────────────────
DEFAULT_COUNTIES = {
    "Alpine":     {"registration": 930,    "turnout": 0.4938, "lean_avg": 0.0388, "lean_lin": 0.0911, "lean_sd": 0.0293, "hist_avg": 0.6221},
    "Amador":     {"registration": 26957,  "turnout": 0.5152, "lean_avg":-0.2465, "lean_lin":-0.2445, "lean_sd": 0.0143, "hist_avg": 0.3481},
    "Calaveras":  {"registration": 32819,  "turnout": 0.4731, "lean_avg":-0.2404, "lean_lin":-0.2516, "lean_sd": 0.0162, "hist_avg": 0.3581},
    "El Dorado":  {"registration": 137636, "turnout": 0.4752, "lean_avg":-0.1784, "lean_lin":-0.1568, "lean_sd": 0.0140, "hist_avg": 0.4079},
    "Inyo":       {"registration": 10856,  "turnout": 0.4943, "lean_avg":-0.1280, "lean_lin":-0.1018, "lean_sd": 0.0122, "hist_avg": 0.4570},
    "Madera":     {"registration": 35735,  "turnout": 0.3708, "lean_avg":-0.2700, "lean_lin":-0.2377, "lean_sd": 0.0182, "hist_avg": 0.3155},
    "Mariposa":   {"registration": 11737,  "turnout": 0.4924, "lean_avg":-0.2083, "lean_lin":-0.2044, "lean_sd": 0.0065, "hist_avg": 0.3772},
    "Merced":     {"registration": 13024,  "turnout": 0.3070, "lean_avg":-0.2673, "lean_lin":-0.2306, "lean_sd": 0.0231, "hist_avg": 0.3182},
    "Mono":       {"registration": 7998,   "turnout": 0.4641, "lean_avg":-0.0251, "lean_lin":-0.0067, "lean_sd": 0.0173, "hist_avg": 0.5628},
    "Nevada":     {"registration": 12272,  "turnout": 0.4980, "lean_avg": 0.1353, "lean_lin": 0.1684, "lean_sd": 0.0181, "hist_avg": 0.7208},
    "Placer":     {"registration": 8560,   "turnout": 0.4615, "lean_avg": 0.1224, "lean_lin": 0.1427, "lean_sd": 0.0113, "hist_avg": 0.7079},
    "Stanislaus": {"registration": 300132, "turnout": 0.3262, "lean_avg":-0.1553, "lean_lin":-0.1543, "lean_sd": 0.0123, "hist_avg": 0.4357},
    "Tuolumne":   {"registration": 35571,  "turnout": 0.4785, "lean_avg":-0.2092, "lean_lin":-0.2039, "lean_sd": 0.0140, "hist_avg": 0.3763},
}

DEFAULT_GLOBAL_SDS = {
    "district_turnout_sd": 0.0823,
    "county_turnout_sd":   0.0773,
    "state_env_sd":        0.0629,
}

WIN_THRESHOLD = 0.50
N_DEFAULT     = 10_000

# ── Initialise session state for editable data ────────────────────────────────
if "county_df" not in st.session_state:
    rows = []
    for cn, d in DEFAULT_COUNTIES.items():
        rows.append({
            "County":       cn,
            "Registration": int(d["registration"]),
            "Turnout (%)":  round(d["turnout"] * 100, 2),
            "Lean Avg (%)": round(d["lean_avg"] * 100, 2),
            "Lean Lin (%)": round(d["lean_lin"] * 100, 2),
            "Lean SD (%)":  round(d["lean_sd"] * 100, 2),
            "Hist Avg (%)": round(d["hist_avg"] * 100, 2),
        })
    st.session_state["county_df"] = pd.DataFrame(rows)

if "global_sds" not in st.session_state:
    st.session_state["global_sds"] = {
        "District Turnout SD (%)": round(DEFAULT_GLOBAL_SDS["district_turnout_sd"] * 100, 2),
        "County Turnout SD (%)":   round(DEFAULT_GLOBAL_SDS["county_turnout_sd"]   * 100, 2),
        "State Env SD (%)":        round(DEFAULT_GLOBAL_SDS["state_env_sd"]        * 100, 2),
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Simulation Parameters")
    n_sims       = st.number_input("Simulations", min_value=1000, max_value=100_000,
                                   value=N_DEFAULT, step=1000)
    forecast_env = st.number_input("Forecast State Environment (%)",
                                   value=59.80, step=0.1) / 100
    lean_method  = st.radio("County Lean Method", ["Average", "Linear"], horizontal=True)

    st.markdown("---")
    st.markdown("### 🔍 Conditional Filters")
    st.caption("County filters use the **vote threshold** below. District win probability always uses 50%.")
    county_vote_threshold = st.number_input("County Vote Threshold (%)", value=50.0, step=0.5) / 100

    county_names_sidebar = list(DEFAULT_COUNTIES.keys())
    county_filters = {}
    for county in county_names_sidebar:
        county_filters[county] = st.selectbox(
            county, ["Ignore", "Over threshold", "Under threshold"],
            key=f"filter_{county}"
        )

    env_lower = st.number_input("State Env Lower (%)", value=0.0,   step=1.0) / 100
    env_upper = st.number_input("State Env Upper (%)", value=100.0, step=1.0) / 100

    run = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ── Pull live parameter values from session state ─────────────────────────────
cdf = st.session_state["county_df"]
gsd = st.session_state["global_sds"]

DISTRICT_TURNOUT_SD = gsd["District Turnout SD (%)"] / 100
COUNTY_TURNOUT_SD   = gsd["County Turnout SD (%)"]   / 100
STATE_ENV_SD        = gsd["State Env SD (%)"]         / 100

# Build COUNTIES dict from the editable dataframe
COUNTIES = {}
for _, row in cdf.iterrows():
    COUNTIES[row["County"]] = {
        "registration": int(row["Registration"]),
        "turnout":      row["Turnout (%)"]  / 100,
        "lean_avg":     row["Lean Avg (%)"] / 100,
        "lean_lin":     row["Lean Lin (%)"] / 100,
        "lean_sd":      row["Lean SD (%)"]  / 100,
        "hist_avg":     row["Hist Avg (%)"] / 100,
    }

# ── Simulation ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running simulations…")
def run_simulation(n, forecast_env, lean_method_str, seed,
                   district_turnout_sd, county_turnout_sd, state_env_sd,
                   counties_frozen):
    """counties_frozen is a tuple-of-tuples so it's hashable for caching."""
    rng = np.random.default_rng(seed)
    n   = int(n)

    counties = {name: dict(fields) for name, fields in counties_frozen}

    env_error              = rng.normal(0, state_env_sd, n)
    sim_env                = forecast_env + env_error
    district_turnout_shift = rng.normal(0, district_turnout_sd, n)

    lean_key         = "lean_avg" if lean_method_str == "Average" else "lean_lin"
    county_shares    = {}
    county_votes     = {}
    county_dem_votes = {}

    for name, data in counties.items():
        turnout_sim = np.clip(
            data["turnout"] + district_turnout_shift + rng.normal(0, county_turnout_sd, n),
            0.01, 1.0
        )
        lean_error = rng.normal(0, data["lean_sd"], n)
        share      = np.clip(sim_env + data[lean_key] + lean_error, 0.0, 1.0)
        votes      = data["registration"] * turnout_sim
        dem_votes  = votes * share

        county_shares[name]    = share
        county_votes[name]     = votes
        county_dem_votes[name] = dem_votes

    total_votes     = sum(county_votes.values())
    total_dem_votes = sum(county_dem_votes.values())
    district_share  = total_dem_votes / total_votes

    return {
        "district_share":    district_share,
        "sim_env":           sim_env,
        "env_error":         env_error,
        "county_shares":     county_shares,
        "county_votes":      county_votes,
        "county_dem_votes":  county_dem_votes,
    }

def counties_to_frozen(counties_dict):
    """Convert COUNTIES dict to a hashable structure for cache keying."""
    return tuple(
        (name, tuple(sorted(data.items())))
        for name, data in counties_dict.items()
    )

if "sim_results" not in st.session_state or run:
    seed = np.random.randint(0, 2**31)
    st.session_state["sim_results"] = run_simulation(
        n_sims, forecast_env, lean_method, seed,
        DISTRICT_TURNOUT_SD, COUNTY_TURNOUT_SD, STATE_ENV_SD,
        counties_to_frozen(COUNTIES)
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
    if county not in county_shares:
        continue
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

# ══════════════════════════════════════════════════════════════════════════════
st.title("🗳️ Election Forecast")
st.caption(
    f"Monte Carlo · {int(n_sims):,} simulations · Lean: {lean_method} · "
    f"District win threshold: 50% (fixed) · County filter threshold: {fmt_pct(county_vote_threshold)}"
)

tab_dash, tab_hood = st.tabs(["📊 Dashboard", "🔧 Under the Hood"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:

    # Summary
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
                st.markdown(f"""
                <div class="stat-card">
                  <div class="label">{label}</div>
                  <div class="value">{val}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribution
    st.markdown('<div class="section-label">Result Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#f7f7f5"); ax.set_facecolor("#f7f7f5")
    ax.hist(ds * 100, bins=60, color="#1a6b3c", alpha=0.75, edgecolor="none")
    ax.axvline(WIN_THRESHOLD * 100, color="#b91c1c", linewidth=1.5, linestyle="--", label="Win threshold (50%)")
    ax.axvline(np.mean(ds) * 100,   color="#0f3d24", linewidth=1.5, linestyle="-",  label=f"Mean {fmt_pct(np.mean(ds))}")
    ax.set_xlabel("Dem Vote Share (%)", fontsize=9); ax.set_ylabel("Simulations", fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(labelsize=8); ax.legend(fontsize=8, framealpha=0)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Threshold table
    st.markdown('<div class="section-label">County Win Probability by Vote Threshold · All Simulations</div>', unsafe_allow_html=True)
    thresholds     = [0.40, 0.45, 0.50]
    county_headers = "".join(f"<th>{cn[:6]}</th>" for cn in county_names)
    rows_html      = ""
    for thr in thresholds:
        district_wp = np.mean(district_share >= WIN_THRESHOLD)
        row  = f'<tr><td>&gt;{fmt_pct(thr, 0)}</td>'
        pc   = prob_class(district_wp)
        row += f'<td class="{pc}">{fmt_pct(district_wp)}</td>'
        for cn in county_names:
            cwp = np.mean(county_shares[cn] >= thr)
            pc  = prob_class(cwp)
            row += f'<td class="{pc}">{fmt_pct(cwp)}</td>'
        row += '</tr>'
        rows_html += row
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>Threshold</th><th>District (50%)</th>{county_headers}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p style="font-size:0.72rem;color:#999;margin-top:0.4rem">
      District column reflects the fixed 50% win threshold. County columns show P(county share &gt; row threshold).
    </p>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # County stats
    st.markdown('<div class="section-label">County Statistics</div>', unsafe_allow_html=True)
    county_rows = ""
    for cn in county_names:
        data   = COUNTIES[cn]
        sim_sh = county_shares[cn]
        mean_s = np.mean(sim_sh); med_s = np.median(sim_sh)
        hist   = data["hist_avg"]; resid = mean_s - hist
        county_rows += (
            f"<tr><td>{cn}</td><td>{fmt_pct(mean_s)}</td><td>{fmt_pct(med_s)}</td>"
            f"<td>{fmt_pct(hist)}</td>"
            f"<td>{'▲' if resid >= 0 else '▼'} {fmt_pct(abs(resid))}</td>"
            f"<td>{fmt_pct(data['lean_sd'])}</td></tr>"
        )
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>County</th><th>Mean Share</th><th>Median Share</th>
      <th>Hist. Average</th><th>Residual</th><th>Lean SD</th></tr></thead>
      <tbody>{county_rows}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Environment bands
    st.markdown('<div class="section-label">Win Probability by State Environment</div>', unsafe_allow_html=True)
    bands = [
        ("≤ 0",       -np.inf, 0.00),
        ("+0 to +5",   0.00,   0.05),
        ("+5 to +8",   0.05,   0.08),
        ("+8 to +10",  0.08,   0.10),
        ("+10 to +12", 0.10,   0.12),
        ("> +12",      0.12,   np.inf),
    ]
    env_rows = ""
    for label, lo, hi in bands:
        band_mask = (env_error >= lo) & (env_error < hi)
        n_band    = band_mask.sum()
        n_wins_b  = int(np.sum(district_share[band_mask] >= WIN_THRESHOLD)) if n_band > 0 else 0
        wp_b      = n_wins_b / n_band if n_band > 0 else 0.0
        pc        = prob_class(wp_b)
        env_rows += (f"<tr><td>{label}</td><td>{n_wins_b:,}</td>"
                     f"<td>{n_band:,}</td><td class='{pc}'>{fmt_pct(wp_b)}</td></tr>")
    st.markdown(f"""
    <table class="styled-table">
      <thead><tr><th>Env. vs Forecast</th><th>Wins</th><th>Simulations</th><th>Win Probability</th></tr></thead>
      <tbody>{env_rows}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Path to victory
    st.markdown('<div class="section-label">Path to Victory</div>', unsafe_allow_html=True)
    win_mask     = district_share >= WIN_THRESHOLD
    n_wins_total = win_mask.sum()
    ptv_rows     = ""
    for cn in county_names:
        county_won = county_shares[cn] >= WIN_THRESHOLD
        p_cw       = (county_won & win_mask).sum() / n_wins_total if n_wins_total > 0 else 0.0
        corr_env   = np.corrcoef(county_shares[cn], sim_env)[0, 1]
        ptv_rows  += (f"<tr><td>{cn}</td><td>{fmt_pct(p_cw)}</td><td>{corr_env:.4f}</td></tr>")

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

    # Turnout analysis
    st.markdown('<div class="section-label">Turnout Analysis</div>', unsafe_allow_html=True)
    fig2, axes = plt.subplots(3, 5, figsize=(14, 6))
    fig2.patch.set_facecolor("#f7f7f5")
    axes_flat = axes.flatten()
    for i, cn in enumerate(county_names):
        sim_turnout = res["county_votes"][cn] / COUNTIES[cn]["registration"]
        ax = axes_flat[i]; ax.set_facecolor("#f7f7f5")
        ax.hist(sim_turnout * 100, bins=30, color="#1a6b3c", alpha=0.7, edgecolor="none")
        ax.axvline(COUNTIES[cn]["turnout"] * 100, color="#b91c1c", linewidth=1.2, linestyle="--")
        ax.set_title(cn, fontsize=7, fontweight="600"); ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.spines[["top","right","left"]].set_visible(False)
    for j in range(len(county_names), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Simulated Turnout by County  (red = forecast)", fontsize=8, color="#555")
    plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()

    st.markdown("---")
    st.caption(
        f"Filtered simulations: {n_filtered:,} of {int(n_sims):,} · "
        f"State env range: {fmt_pct(env_lower)} – {fmt_pct(env_upper)} · "
        f"County filters active: {sum(1 for v in county_filters.values() if v != 'Ignore')}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UNDER THE HOOD
# ══════════════════════════════════════════════════════════════════════════════
with tab_hood:

    st.markdown("## How the model works")
    st.markdown(
        "This is a county-level Monte Carlo simulation. Each of the "
        f"**{int(n_sims):,} simulations** independently draws random values for the "
        "state environment, county turnouts, and county vote shares, then rolls them up "
        "to a single district result. The win probability is the share of simulations "
        "where the Democrat clears **50%** of the total district vote."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Editable global SDs ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Global Standard Deviations</div>', unsafe_allow_html=True)
    st.caption("Edit any value and hit **Run Simulation** in the sidebar to apply.")

    sd_df = pd.DataFrame([
        {"Parameter": k, "Value (%)": v}
        for k, v in st.session_state["global_sds"].items()
    ])

    edited_sd_df = st.data_editor(
        sd_df,
        use_container_width=True,
        hide_index=True,
        disabled=["Parameter"],
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter", width="large"),
            "Value (%)": st.column_config.NumberColumn("Value (%)", min_value=0.0, max_value=50.0,
                                                        step=0.01, format="%.2f"),
        },
        key="sd_editor",
    )

    # Persist edits back to session state
    for _, row in edited_sd_df.iterrows():
        st.session_state["global_sds"][row["Parameter"]] = row["Value (%)"]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Editable county data ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">County Parameters</div>', unsafe_allow_html=True)
    st.caption(
        "Edit registration, turnout, lean values, or SDs as new data comes in. "
        "Hit **Run Simulation** in the sidebar to apply. "
        "All percentage fields are in plain % (e.g. 49.38, not 0.4938)."
    )

    edited_county_df = st.data_editor(
        st.session_state["county_df"],
        use_container_width=True,
        hide_index=True,
        disabled=["County"],
        column_config={
            "County":       st.column_config.TextColumn("County", width="medium"),
            "Registration": st.column_config.NumberColumn("Registration", min_value=0, step=1, format="%d"),
            "Turnout (%)":  st.column_config.NumberColumn("Turnout (%)",  min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "Lean Avg (%)": st.column_config.NumberColumn("Lean Avg (%)", min_value=-100.0, max_value=100.0, step=0.01, format="%.2f"),
            "Lean Lin (%)": st.column_config.NumberColumn("Lean Lin (%)", min_value=-100.0, max_value=100.0, step=0.01, format="%.2f"),
            "Lean SD (%)":  st.column_config.NumberColumn("Lean SD (%)",  min_value=0.0, max_value=50.0,  step=0.01, format="%.2f"),
            "Hist Avg (%)": st.column_config.NumberColumn("Hist Avg (%)", min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        },
        key="county_editor",
    )

    # Persist edits
    st.session_state["county_df"] = edited_county_df

    col_reset_c, col_reset_sd, _ = st.columns([1, 1, 3])
    with col_reset_c:
        if st.button("↩ Reset county data to defaults"):
            rows = []
            for cn, d in DEFAULT_COUNTIES.items():
                rows.append({
                    "County":       cn,
                    "Registration": int(d["registration"]),
                    "Turnout (%)":  round(d["turnout"] * 100, 2),
                    "Lean Avg (%)": round(d["lean_avg"] * 100, 2),
                    "Lean Lin (%)": round(d["lean_lin"] * 100, 2),
                    "Lean SD (%)":  round(d["lean_sd"] * 100, 2),
                    "Hist Avg (%)": round(d["hist_avg"] * 100, 2),
                })
            st.session_state["county_df"] = pd.DataFrame(rows)
            st.rerun()
    with col_reset_sd:
        if st.button("↩ Reset SDs to defaults"):
            st.session_state["global_sds"] = {
                "District Turnout SD (%)": round(DEFAULT_GLOBAL_SDS["district_turnout_sd"] * 100, 2),
                "County Turnout SD (%)":   round(DEFAULT_GLOBAL_SDS["county_turnout_sd"]   * 100, 2),
                "State Env SD (%)":        round(DEFAULT_GLOBAL_SDS["state_env_sd"]        * 100, 2),
            }
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model steps ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Model Steps</div>', unsafe_allow_html=True)
    steps = [
        ("1. State Environment",
         "A single environment value is drawn for each simulation from a normal distribution "
         "centred on the forecast. This represents the overall partisan environment.",
         f"<code>sim_env = N({fmt_pct(forecast_env)}, {fmt_pct(STATE_ENV_SD)} SD)</code>"),
        ("2. District Turnout Shock",
         "A single district-wide turnout shift is drawn per simulation and applied equally "
         "to every county, capturing correlated turnout swings.",
         f"<code>district_shift = N(0, {fmt_pct(DISTRICT_TURNOUT_SD)} SD)</code>"),
        ("3. County Turnout",
         "Each county gets its own idiosyncratic turnout noise on top of the district shift. "
         "Simulated turnout is clipped to [1%, 100%].",
         f"<code>county_turnout = forecast_turnout + district_shift + N(0, {fmt_pct(COUNTY_TURNOUT_SD)} SD)</code>"),
        ("4. County Vote Share",
         f"Each county's Democratic share = simulated state environment + county lean "
         f"({'average' if lean_method == 'Average' else 'linear regression'} method) "
         f"+ county-specific lean error.",
         "<code>county_share = sim_env + county_lean + N(0, county_lean_SD)</code>"),
        ("5. Aggregation",
         "Vote totals = registration × turnout per county. Dem votes = total votes × county share. "
         "Summed across counties to produce the district share.",
         "<code>district_share = Σ(dem_votes) / Σ(total_votes)</code>"),
        ("6. Win Probability",
         "Win probability = proportion of simulations where district_share ≥ 50%. "
         "Conditional filters narrow the simulation set without changing the underlying draws.",
         "<code>P(win) = mean(district_share ≥ 50%)</code>"),
    ]
    for title, body, formula in steps:
        st.markdown(f"""
        <div class="model-box">
          <h4>{title}</h4><p>{body}</p><p>{formula}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Lean comparison chart ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Lean Method Comparison</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    fig3.patch.set_facecolor("#f7f7f5"); ax3.set_facecolor("#f7f7f5")
    x    = np.arange(len(county_names)); w = 0.35
    avgs = [COUNTIES[cn]["lean_avg"] * 100 for cn in county_names]
    lins = [COUNTIES[cn]["lean_lin"] * 100 for cn in county_names]
    ax3.bar(x - w/2, avgs, w, label="Average lean", color="#1a6b3c", alpha=0.8)
    ax3.bar(x + w/2, lins, w, label="Linear lean",  color="#0f3d24", alpha=0.5)
    ax3.axhline(0, color="#ccc", linewidth=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels(county_names, rotation=40, ha="right", fontsize=7)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.set_ylabel("Lean vs State Env (%)", fontsize=8)
    ax3.legend(fontsize=8, framealpha=0); ax3.spines[["top","right","left"]].set_visible(False)
    ax3.tick_params(labelsize=7); plt.tight_layout()
    st.pyplot(fig3, use_container_width=True); plt.close()
