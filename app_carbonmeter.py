import io
import os
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# -----------------------------------------------------------
# Basic page config
# -----------------------------------------------------------
st.set_page_config(
    page_title="CarbonMindful: Smart Carbon Footprint Calculator for AI Model Training",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------
# Custom CSS for full black background & readable text
# -----------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
    }
    /* Make markdown text a bit larger and readable */
    .stMarkdown, p, li {
        font-size: 0.95rem !important;
    }
    /* Tighten padding a bit */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------

EMISSION_FACTORS = {
    "CAN": 0.12,  # Canada
    "MEX": 0.43,  # Mexico
    "KOR": 0.45,  # South Korea
    "MNG": 0.75,  # Mongolia
}

REGION_NAME = {
    "CAN": "Canada",
    "MEX": "Mexico",
    "KOR": "South Korea",
    "MNG": "Mongolia",
}

REGION_COLORS = {
    "CAN": "#00c853",  # bright green
    "MEX": "#ff9100",  # orange
    "KOR": "#00b0ff",  # blue
    "MNG": "#e040fb",  # purple
}


def load_run_bundle_from_zip(uploaded_file) -> Tuple[Optional[Path], Optional[Dict[str, Any]], Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Extracts an uploaded ZIP file into a temp directory, finds a folder
    containing summary.json, and loads:
      - summary (dict)
      - epochs_df (if epochs.csv exists)
      - samples_df (if samples.csv exists)
    Returns (run_dir, summary, epochs_df, samples_df, status_msg)
    """
    if uploaded_file is None:
        return None, None, None, None, "No file uploaded."

    # Create a temp directory
    temp_dir = Path(tempfile.mkdtemp())
    zip_path = temp_dir / "bundle.zip"

    # Save uploaded file to disk
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract zip
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)
    except zipfile.BadZipFile:
        return None, None, None, None, "Uploaded file is not a valid ZIP archive."

    # Find summary.json anywhere under temp_dir
    run_dir = None
    summary = None
    for root, dirs, files in os.walk(temp_dir):
        if "summary.json" in files:
            run_dir = Path(root)
            try:
                summary = json.loads((run_dir / "summary.json").read_text())
            except Exception as e:
                return None, None, None, None, f"Error reading summary.json: {e}"
            break

    if run_dir is None or summary is None:
        return None, None, None, None, "Could not find summary.json in the uploaded ZIP."

    # Try loading optional CSVs
    epochs_df = None
    samples_df = None

    epochs_path = run_dir / "epochs.csv"
    if epochs_path.exists():
        try:
            epochs_df = pd.read_csv(epochs_path)
        except Exception:
            epochs_df = None

    samples_path = run_dir / "samples.csv"
    if samples_path.exists():
        try:
            samples_df = pd.read_csv(samples_path)
        except Exception:
            samples_df = None

    return run_dir, summary, epochs_df, samples_df, "Bundle loaded successfully."


def compute_region_scenarios(
    energy_kwh: float,
    baseline_pue: float,
    pue_scenarios: Optional[list] = None,
) -> pd.DataFrame:
    """
    Given total energy in kWh, compute emissions (kg CO2e)
    for CAN/MEX/KOR/MNG for one or more PUE values.
    Returns a tidy DataFrame.
    """
    if pue_scenarios is None:
        pue_scenarios = [baseline_pue]

    rows = []
    for iso, factor in EMISSION_FACTORS.items():
        for pue in pue_scenarios:
            emissions_kg = energy_kwh * pue * factor
            rows.append(
                {
                    "country_iso": iso,
                    "country": REGION_NAME[iso],
                    "pue": float(pue),
                    "grid_factor_kg_per_kwh": factor,
                    "emissions_kg": emissions_kg,
                }
            )

    df = pd.DataFrame(rows).sort_values(["country_iso", "pue"]).reset_index(drop=True)
    return df


def plot_bar_by_country(df: pd.DataFrame, title: str):
    """
    Simple bar chart: emissions by country (single PUE).
    df must have columns: country_iso, country, emissions_kg
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    countries = df["country"].tolist()
    values = df["emissions_kg"].tolist()
    isos = df["country_iso"].tolist()

    colors = [REGION_COLORS.get(iso, "#ffffff") for iso in isos]

    ax.bar(countries, values, color=colors)

    ax.set_title(title, color="white", fontsize=12)
    ax.set_ylabel("Emissions (kg CO₂e)", color="white")
    ax.set_xlabel("Country", color="white")
    ax.tick_params(axis="x", colors="white", rotation=20)
    ax.tick_params(axis="y", colors="white")

    # Make spines visible against black
    for spine in ax.spines.values():
        spine.set_color("white")

    # Add value labels
    for i, v in enumerate(values):
        ax.text(
            i,
            v,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
        )

    plt.tight_layout()
    st.pyplot(fig)


def plot_pue_sensitivity(df: pd.DataFrame, title: str):
    """
    Grouped bar chart: for each country, emissions for multiple PUE values.
    df must have columns: country_iso, country, pue, emissions_kg
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    countries = df["country"].unique().tolist()
    pue_values = sorted(df["pue"].unique().tolist())

    x = np.arange(len(countries))
    width = 0.8 / max(1, len(pue_values))

    for idx, pue in enumerate(pue_values):
        df_sub = df[df["pue"] == pue]
        heights = []
        for c in countries:
            row = df_sub[df_sub["country"] == c].iloc[0]
            heights.append(row["emissions_kg"])

        bar_positions = x + (idx - (len(pue_values) - 1) / 2) * width
        ax.bar(bar_positions, heights, width=width, label=f"PUE={pue}", alpha=0.8)

    ax.set_title(title, color="white", fontsize=12)
    ax.set_ylabel("Emissions (kg CO₂e)", color="white")
    ax.set_xlabel("Country", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(countries, color="white", rotation=20)
    ax.tick_params(axis="y", colors="white")

    for spine in ax.spines.values():
        spine.set_color("white")

    legend = ax.legend(facecolor="#111111", edgecolor="white")
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    st.pyplot(fig)


def plot_epochs_emissions_and_accuracy(epochs_df: pd.DataFrame, total_emissions_kg: Optional[float]):
    """
    Shows:
      - Emissions per epoch
      - Cumulative emissions
      - Accuracy vs epoch (if val_acc exists)
    """
    if "epoch" not in epochs_df.columns:
        st.info("epochs.csv does not have 'epoch' column; skipping epoch plots.")
        return

    emissions_col = "emissions_kg" if "emissions_kg" in epochs_df.columns else None
    val_acc_col = "val_acc" if "val_acc" in epochs_df.columns else None

    epochs_df = epochs_df.copy()
    epochs_df["epoch"] = epochs_df["epoch"].astype(int)

    cols = st.columns(2)

    # Emissions per epoch
    with cols[0]:
        if emissions_col:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#000000")
            ax.set_facecolor("#000000")

            vals = epochs_df[emissions_col].fillna(0.0)
            ax.bar(epochs_df["epoch"], vals, color="#00e676")
            ax.set_title("Emissions per epoch", color="white")
            ax.set_xlabel("Epoch", color="white")
            ax.set_ylabel("kg CO₂e", color="white")
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")

            for e, v in zip(epochs_df["epoch"], vals):
                ax.text(e, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7, color="white")

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No emissions_kg column in epochs.csv; per-epoch emissions plot skipped.")

    # Cumulative emissions
    with cols[1]:
        if emissions_col:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#000000")
            ax.set_facecolor("#000000")

            vals = epochs_df[emissions_col].fillna(0.0)
            cum = vals.cumsum()
            ax.plot(epochs_df["epoch"], cum, marker="o", color="#ffea00")
            ax.set_title("Cumulative emissions", color="white")
            ax.set_xlabel("Epoch", color="white")
            ax.set_ylabel("Cumulative kg CO₂e", color="white")
            ax.tick_params(axis="x", colors="white")
            ax.tick_params(axis="y", colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No emissions_kg column in epochs.csv; cumulative emissions plot skipped.")

    # Accuracy vs epoch
    if val_acc_col:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#000000")

        acc_vals = epochs_df[val_acc_col].fillna(0.0) * 100.0
        ax.plot(epochs_df["epoch"], acc_vals, marker="o", color="#40c4ff")
        ax.set_title("Validation accuracy vs epoch", color="white")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("Accuracy (%)", color="white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        plt.tight_layout()
        st.pyplot(fig)


def plot_utilization(samples_df: pd.DataFrame):
    """
    Plots GPU utilization over time if present, else CPU utilization.
    """
    if "timestamp" not in samples_df.columns:
        st.info("samples.csv is missing timestamp column; utilization plot skipped.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    t0 = samples_df["timestamp"].iloc[0]
    tmin = (samples_df["timestamp"] - t0) / 60.0

    if "gpu_util_pct" in samples_df.columns and samples_df["gpu_util_pct"].notna().any():
        y = samples_df["gpu_util_pct"].fillna(0.0)
        label = "GPU utilization (%)"
        color = "#ff5252"
    else:
        y = samples_df["cpu_util_pct"].fillna(0.0)
        label = "CPU utilization (%)"
        color = "#00e676"

    ax.plot(tmin, y, color=color)
    ax.set_title("Hardware utilization over time", color="white")
    ax.set_xlabel("Time (minutes)", color="white")
    ax.set_ylabel(label, color="white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------

st.sidebar.title("Smart Carbon Footprint Calculator")

st.sidebar.markdown(
    "Upload a **run bundle** (.zip) generated by your training script.\n\n"
    "The bundle should contain at least:\n"
    "- `summary.json` (required)\n"
    "- `epochs.csv` (optional, for accuracy & per-epoch metrics)\n"
    "- `samples.csv` (optional, for utilization plots)"
)

st.sidebar.markdown("---")

baseline_option = st.sidebar.radio(
    "Baseline interpretation",
    [
        "Use original measurement from summary.json",
        "Override baseline region/PUE",
    ],
    help=(
        "The baseline is the reference footprint used for interpretation and for the "
        "efficiency metric (kg CO₂e per 1% accuracy). "
        "If you override it, the app recomputes a hypothetical baseline emissions value "
        "using your chosen region and PUE."
    ),
)

baseline_region = st.sidebar.selectbox(
    "If overriding, assume this run happened in:",
    options=["Online (auto)", "Canada (CAN)", "Mexico (MEX)", "South Korea (KOR)", "Mongolia (MNG)"],
    help=(
        "This does NOT change your measured energy (kWh). It only changes how that energy is "
        "translated into emissions for the baseline scenario, by assuming a different location "
        "and grid factor."
    ),
)

baseline_pue = st.sidebar.slider(
    "Baseline PUE (Power Usage Effectiveness)",
    min_value=1.0,
    max_value=2.0,
    value=1.2,
    step=0.05,
    help=(
        "PUE represents how efficient the data center is. "
        "A PUE of 1.2 means that for every 1 kWh used by your hardware, "
        "0.2 kWh are spent on cooling/overhead."
    ),
)

pue_sensitivity_values = st.sidebar.multiselect(
    "Additional PUE scenarios for sensitivity plots",
    options=[1.2, 1.4, 1.6],
    default=[1.2, 1.6],
    help="These will be used to compare how emissions change at different PUE values.",
)

st.sidebar.markdown("---")
show_epoch_plots = st.sidebar.checkbox("Show per-epoch plots (if available)", value=True)
show_util_plots = st.sidebar.checkbox("Show utilization plot (if samples.csv available)", value=True)


# -----------------------------------------------------------
# Main layout
# -----------------------------------------------------------

st.title("Smart Carbon Footprint Calculator")

st.markdown(
    """
This app analyzes **completed AI training runs** and estimates their carbon footprint,
then shows how that footprint would change in different regions and data-center conditions.

### How to make the most of this platform

1. **Generate a run bundle** from your own training code using a logger that writes:
   - `summary.json` (required),
   - optionally `epochs.csv` and `samples.csv`.
2. **Upload the bundle** here to see:
   - your baseline energy and emissions,
   - how emissions change if the same run happened in **Canada, Mexico, South Korea, or Mongolia**,
   - optional links between **accuracy** and **emissions**.
3. Use the **baseline override controls** in the sidebar to explore:
   - how different locations and PUE assumptions change the footprint,
   - how sensitive your conclusions are to infrastructure choices.
"""
)

st.markdown("### 1. Upload your Run Bundle")

uploaded_file = st.file_uploader("Upload a ZIP file containing one run folder", type=["zip"])

# Detailed bundle format explanation
with st.expander("What should my run bundle contain? (Click for details)"):
    st.markdown("""
**Required file – `summary.json`**

This JSON file summarizes one completed training run. It must include:

- `project_name` *(string)* – name of your project (e.g., `"carbon_calculator_mvp"`).
- `run_name` *(string)* – identifier for this run (e.g., `"mnist_demo_kor"`).
- `tracker_mode` *(string)* – `"online"` or `"offline"`, depending on how CodeCarbon was used.
- `country_iso_code` *(string or null)* – ISO code if `tracker_mode` is `"offline"` (e.g., `"KOR"`), else `null`.
- `pue` *(number)* – Power Usage Effectiveness assumed for the data center (e.g., `1.20`).
- `duration_s` *(number)* – total run time in seconds.
- `total_energy_kwh` *(number)* – total electrical energy consumed by the run, in kWh.
- `total_emissions_kg` *(number)* – total CO₂-equivalent emissions (kg CO₂e).

Optional but recommended:

- `epochs` *(int)* – number of epochs in this run.
- `tags` *(object)* – free metadata (e.g., dataset name, model name, framework).
""")

    st.markdown("""
**Optional file – `epochs.csv`**

Per-epoch metrics. Recommended columns:

- `epoch` – epoch index (integer).
- `duration_s` – duration of this epoch in seconds.
- `train_loss`, `train_acc` – training loss/accuracy.
- `val_loss`, `val_acc` – validation loss/accuracy.
- `energy_kwh` – energy attributed to this epoch (optional).
- `emissions_kg` – emissions attributed to this epoch (optional).

This file enables accuracy-related plots and the “kg CO₂e per 1% accuracy” metric.
""")

    st.markdown("""
**Optional file – `samples.csv`**

Time-series samples of hardware usage. Typical columns:

- `timestamp` – UNIX time (seconds).
- `cpu_util_pct` – CPU utilization percentage.
- `cpu_mem_gb_used`, `cpu_mem_gb_total` – CPU memory usage and total (GB).
- `gpu_util_pct` – GPU utilization percentage (if GPU used).
- `gpu_mem_gb_used`, `gpu_mem_gb_total` – GPU memory usage and total (GB).
- `gpu_name`, `gpu_total_devices` – GPU name and count.

This file enables the hardware utilization plot.
""")

if not uploaded_file:
    st.info(
        "Waiting for a run bundle. Please upload a `.zip` file that contains a folder with "
        "`summary.json` (required) and optionally `epochs.csv`, `samples.csv`."
    )
    st.stop()

run_dir, summary, epochs_df, samples_df, status_msg = load_run_bundle_from_zip(uploaded_file)

if summary is None:
    st.error(f"❌ {status_msg}")
    st.stop()
else:
    st.success(f"✅ {status_msg}")

# -----------------------------------------------------------
# Bundle health summary
# -----------------------------------------------------------

st.markdown("### 2. Bundle Health & Metadata")

cols_meta = st.columns(3)
with cols_meta[0]:
    st.metric("Project", summary.get("project_name", "N/A"))
    st.metric("Run name", summary.get("run_name", "N/A"))

with cols_meta[1]:
    tracker_mode = summary.get("tracker_mode", "N/A")
    iso = summary.get("country_iso_code", None)
    if tracker_mode == "online":
        region_str = "Online (auto-detected)"
    else:
        region_str = iso or "N/A"
    st.metric("Tracker mode", tracker_mode)
    st.metric("Measured region", region_str)

with cols_meta[2]:
    dur_s = summary.get("duration_s", None)
    if dur_s is not None:
        minutes = dur_s / 60.0
        dur_str = f"{minutes:.1f} min"
    else:
        dur_str = "N/A"
    st.metric("Duration", dur_str)
    st.metric("PUE (from run)", summary.get("pue", "N/A"))

with st.expander("Files detected in this bundle", expanded=False):
    st.write(f"Run directory (inside ZIP): `{run_dir}`")
    st.write(f"- `summary.json`: ✅")
    st.write(f"- `epochs.csv`: {'✅' if epochs_df is not None else '❌'}")
    st.write(f"- `samples.csv`: {'✅' if samples_df is not None else '❌'}")

st.markdown("""
**How to interpret this section**

- This tells you *what run you are looking at* (project, run name) and under which conditions it was originally measured.
- If the tracker mode is **online**, the emissions used the location and cloud provider detected by CodeCarbon.
- If the tracker mode is **offline**, emissions were computed assuming the specified `country_iso_code` and PUE.
- The rest of the app will treat the **measured energy (kWh)** as fixed, and explore how emissions change under different regional assumptions.
""")

# -----------------------------------------------------------
# Baseline footprint
# -----------------------------------------------------------

st.markdown("### 3. Baseline Footprint (Original Run or Overridden)")

total_energy_kwh = summary.get("total_energy_kwh", None)
total_emissions_kg = summary.get("total_emissions_kg", None)

if total_energy_kwh is None:
    st.error("summary.json is missing 'total_energy_kwh'; cannot compute footprints.")
    st.stop()

baseline_info_lines = []

if baseline_option == "Use original measurement from summary.json":
    baseline_emissions_kg = total_emissions_kg
    baseline_pue_effective = summary.get("pue", baseline_pue)
    baseline_info_lines.append("Using emissions reported in summary.json as baseline.")
else:
    # Override baseline: recompute baseline emissions using selected region/PUE
    iso_override = None
    if baseline_region == "Canada (CAN)":
        iso_override = "CAN"
    elif baseline_region == "Mexico (MEX)":
        iso_override = "MEX"
    elif baseline_region == "South Korea (KOR)":
        iso_override = "KOR"
    elif baseline_region == "Mongolia (MNG)":
        iso_override = "MNG"

    if iso_override is None:
        # Online (auto) override: use average of our four demo factors
        avg_factor = np.mean(list(EMISSION_FACTORS.values()))
        baseline_emissions_kg = total_energy_kwh * baseline_pue * avg_factor
        baseline_info_lines.append(
            "Baseline overridden: treating this run as if executed in an 'online/average' grid "
            f"with PUE={baseline_pue:.2f}."
        )
    else:
        factor = EMISSION_FACTORS[iso_override]
        baseline_emissions_kg = total_energy_kwh * baseline_pue * factor
        baseline_info_lines.append(
            f"Baseline overridden: treating this run as if executed in {REGION_NAME[iso_override]} "
            f"(factor={factor} kg CO₂e/kWh, PUE={baseline_pue:.2f})."
        )
    baseline_pue_effective = baseline_pue

col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    st.metric("Total energy used", f"{total_energy_kwh:.6f} kWh")
with col_b2:
    if baseline_emissions_kg is not None:
        st.metric("Baseline emissions", f"{baseline_emissions_kg:.6f} kg CO₂e")
    else:
        st.metric("Baseline emissions", "N/A")
with col_b3:
    st.metric("Baseline PUE used", f"{baseline_pue_effective:.2f}")

for line in baseline_info_lines:
    st.markdown(f"- {line}")

st.markdown("""
**How to read this section**

- **Total energy used** is the fixed physical energy that your training run consumed.
- **Baseline emissions** convert that energy into CO₂e using an emission factor and PUE.
- If you **do not override** the baseline, the app uses the emissions directly from `summary.json`.
- If you **override** the baseline, the app keeps energy the same but recomputes emissions as if the run had been executed in a different region and/or PUE.

You can use this to:

- Compare your original footprint with a **what-if scenario** (e.g., *what if I ran this in Canada with a PUE of 1.2?*).
- Understand how sensitive your footprint is to **infrastructure choices**, not just your model.
""")

# -----------------------------------------------------------
# Regional scenarios
# -----------------------------------------------------------

st.markdown("### 4. Regional Comparison (Canada, Mexico, South Korea, Mongolia)")

# Build PUE scenario list (baseline + selected extras)
pue_scenarios = [baseline_pue_effective]
for v in pue_sensitivity_values:
    if v not in pue_scenarios:
        pue_scenarios.append(v)
pue_scenarios = sorted(pue_scenarios)

df_regions = compute_region_scenarios(total_energy_kwh, baseline_pue_effective, pue_scenarios=pue_scenarios)

st.markdown("""
This table assumes your **measured energy (kWh)** is fixed, and asks:

> *“What would the emissions be if this exact same energy demand were supplied in different countries and at different PUE values?”*

- `grid_factor_kg_per_kwh` is the assumed average carbon intensity of the electricity grid.
- `pue` is the data center efficiency.
- `emissions_kg` is the resulting CO₂e footprint for that scenario.

Changing the **baseline PUE** in the sidebar changes the PUE used here for the baseline scenario, and therefore scales emissions up or down.
""")

st.write("**Scenario table:** Same energy, different grids and PUE values.")
st.dataframe(df_regions, use_container_width=True)

# Single-PUE view (baseline only)
df_baseline_only = df_regions[df_regions["pue"] == baseline_pue_effective].copy()
plot_bar_by_country(df_baseline_only, f"Emissions by country at PUE={baseline_pue_effective:.2f}")

st.markdown("""
**Interpretation tip**

- The **height of each bar** shows how much CO₂e would be emitted if your run used the same energy in that country and at the given PUE.
- Lower bars are better from a climate perspective.
- Countries with cleaner grids and/or better PUE will show **substantially lower emissions** for the same training run.
""")

# Multi-PUE sensitivity plot (if more than one PUE)
if len(pue_scenarios) > 1:
    plot_pue_sensitivity(df_regions, "PUE sensitivity across regions (same energy, different efficiencies)")
    st.markdown("""
**PUE sensitivity**

This plot answers: *“If the grid stays the same, how does changing the data-center efficiency (PUE) alone affect emissions?”*

- Moving from **PUE 1.6 → 1.2** reduces overhead energy and therefore emissions.
- For the same country, bars at lower PUEs should be **consistently lower**.
""")

# Quick insights block
df_baseline_only = df_regions[df_regions["pue"] == baseline_pue_effective].copy()
best_row = df_baseline_only.loc[df_baseline_only["emissions_kg"].idxmin()]
worst_row = df_baseline_only.loc[df_baseline_only["emissions_kg"].idxmax()]
reduction_pct = 100.0 * (worst_row["emissions_kg"] - best_row["emissions_kg"]) / worst_row["emissions_kg"]

st.markdown("#### Quick Insights")

st.markdown(
    f"""
- **Lowest-emission region (at PUE={baseline_pue_effective:.2f}):** {best_row['country']}  
  → {best_row['emissions_kg']:.4f} kg CO₂e
- **Highest-emission region:** {worst_row['country']}  
  → {worst_row['emissions_kg']:.4f} kg CO₂e
- **Relative difference:** Running this same training job in {best_row['country']} instead of {worst_row['country']}
  would reduce the footprint by about **{reduction_pct:.1f}%**.
"""
)

st.markdown("""
You can use these insights in reports or a thesis as a concrete statement, for example:

> “For this model and dataset, choosing a low-carbon region can reduce training emissions by roughly X% without changing the code.”
""")

# -----------------------------------------------------------
# Model Performance & Efficiency
# -----------------------------------------------------------

if epochs_df is not None and "val_acc" in epochs_df.columns:
    st.markdown("### 5. Model Performance & Carbon Efficiency")

    df_acc = epochs_df[epochs_df["val_acc"].notna()]
    final_acc = None
    if not df_acc.empty:
        final_acc = float(df_acc["val_acc"].tail(1).values[0])

    if final_acc is not None and baseline_emissions_kg is not None and final_acc > 0:
        kg_per_pct = baseline_emissions_kg / (final_acc * 100.0)
    else:
        kg_per_pct = None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Final validation accuracy", f"{final_acc * 100.0:.2f} %" if final_acc is not None else "N/A")
    with c2:
        if baseline_emissions_kg is not None:
            st.metric("Total emissions (baseline)", f"{baseline_emissions_kg:.6f} kg CO₂e")
        else:
            st.metric("Total emissions (baseline)", "N/A")
    with c3:
        if kg_per_pct is not None:
            st.metric("kg CO₂e per 1% accuracy", f"{kg_per_pct:.6f} kg/percentage point")
        else:
            st.metric("kg CO₂e per 1% accuracy", "N/A")

    st.markdown("""
This metric helps you evaluate **diminishing returns**:

- A **smaller** value of “kg CO₂e per 1% accuracy” means the run is more carbon-efficient.
- Comparing different runs with this metric shows where extra training or larger models give **tiny accuracy gains** for a **large extra footprint**.
- In a thesis or report, you can say:  
  *“In this configuration, each additional percentage point of validation accuracy costs about X kg of CO₂e.”*
""")

    if show_epoch_plots:
        plot_epochs_emissions_and_accuracy(epochs_df, baseline_emissions_kg)
else:
    st.markdown("### 5. Model Performance & Carbon Efficiency")
    st.info(
        "No usable `epochs.csv` with `val_acc` found in the bundle. "
        "Accuracy-based efficiency metrics are not available for this run."
    )

# -----------------------------------------------------------
# Utilization plot
# -----------------------------------------------------------

st.markdown("### 6. Hardware Utilization Over Time")

st.markdown("""
This plot shows how busy your CPU or GPU was during the run.

- Flat, low utilization (e.g., < 40% most of the time) suggests **wasted capacity** and potentially avoidable emissions.
- Higher average utilization suggests that hardware was used more efficiently for the same training task.
- You can use this to argue for optimizing **batch size, data pipeline, or model placement** to reduce idle time.
""")

if samples_df is not None and show_util_plots:
    plot_utilization(samples_df)
else:
    st.info(
        "No `samples.csv` found or utilization plotting disabled. "
        "Upload a bundle with samples.csv to visualize CPU/GPU utilization."
    )
