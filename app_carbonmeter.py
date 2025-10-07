

# app_carbonmeter.py
# Carbon Footprint Calculator for AI Models — Daniel Ojeda Rosales
# Streamlit app: dark UI, white-background plots, preload demo ZIPs from sample_runs/

import json
import io, zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------- Page setup & dark shell -------------------------
st.set_page_config(
    page_title="Carbon Footprint Calculator for AI Models — Daniel Ojeda Rosales",
    page_icon="🌍",
    layout="wide",
)

# Dark background shell (keep plots white)
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    h1, h2, h3, h4, h5, h6, p, li, div, span, label { color: #e6e6e6 !important; }
    .stMetric { background: rgba(255,255,255,0.04); border-radius: 14px; padding: 8px 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Carbon Footprint Calculator for AI Models — Daniel Ojeda Rosales")
st.caption("Visualize energy & emissions from ML training runs (CodeCarbon-compatible).")

# ------------------------- Constants & mappings ----------------------------
REGION_NAME = {
    "KOR": "South Korea",
    "MEX": "Mexico",
    "CAN": "Canada",
    "FRA": "France",
    "MNG": "Mongolia",
}

# Approximate national-average grid intensities (kg CO2e / kWh)
EMISSION_FACTORS = {
    "KOR": 0.45,
    "MEX": 0.43,
    "CAN": 0.12,
    "FRA": 0.05,
    "MNG": 0.75,
}

REQUIRED_FILES = {"summary.json", "epochs.csv", "samples.csv", "emissions.csv"}

# ------------------------- Robust ZIP helpers ------------------------------
def _find_run_root(extracted_dir: Path) -> Path:
    """
    Find the directory that contains all required files, even if nested.
    """
    # Check current level
    names_here = {p.name for p in extracted_dir.iterdir() if p.is_file()}
    if REQUIRED_FILES.issubset(names_here):
        return extracted_dir

    # Search recursively for a folder containing the full set
    for sj in extracted_dir.rglob("summary.json"):
        candidate = sj.parent
        if all((candidate / f).exists() for f in REQUIRED_FILES):
            return candidate

    raise FileNotFoundError(
        "Missing one or more required files (summary.json, epochs.csv, samples.csv, emissions.csv) in the uploaded ZIP."
    )

def extract_zip_to_tmp(uploaded_file) -> Path:
    """
    Extract a user-uploaded ZIP (UploadedFile) into /tmp and return the run root dir.
    """
    out_dir = Path("/tmp") / f"run_{Path(uploaded_file.name).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    zbytes = io.BytesIO(uploaded_file.getvalue())
    with zipfile.ZipFile(zbytes, "r") as zf:
        zf.extractall(out_dir)
    return _find_run_root(out_dir)

def extract_zip_path_to_tmp(zip_path: Path) -> Path:
    """
    Extract a repository-shipped ZIP (by path) into /tmp and return the run root dir.
    """
    out_dir = Path("/tmp") / f"run_{zip_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(out_dir)
    return _find_run_root(out_dir)

def list_repo_zip_paths() -> List[Path]:
    """
    Find *.zip under sample_runs/, checking both file dir and CWD (Streamlit can differ).
    """
    roots = [Path(__file__).resolve().parent / "sample_runs", Path.cwd() / "sample_runs"]
    zips: List[Path] = []
    for r in roots:
        try:
            if r.exists():
                zips.extend(sorted(r.glob("*.zip")))
        except Exception:
            pass
    return zips

# ------------------------- Run loading & parsing ---------------------------
def load_run_dir(run_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Read required files from a run directory.
    Returns: (summary_dict, epochs_df, samples_df, emissions_df|None)
    """
    summary = json.loads((run_dir / "summary.json").read_text())
    epochs = pd.read_csv(run_dir / "epochs.csv")
    samples = pd.read_csv(run_dir / "samples.csv")
    emissions_df = None
    em_csv = run_dir / "emissions.csv"
    if em_csv.exists():
        emissions_df = pd.read_csv(em_csv)
    return summary, epochs, samples, emissions_df

def run_label(summary: Dict[str, Any]) -> str:
    mode = summary.get("tracker_mode", "unknown")
    iso = summary.get("country_iso_code")
    pue = summary.get("pue")
    name = summary.get("run_name", "run")
    if mode == "offline" and iso:
        region = REGION_NAME.get(iso, iso)
        base = f"{name} — Offline ({region})"
    elif mode == "online":
        base = f"{name} — Online (auto region)"
    else:
        base = f"{name} — {mode}"
    if pue:
        base += f" · PUE={float(pue):.2f}"
    return base

def measured_kwh(summary: Dict[str, Any], emissions_df: Optional[pd.DataFrame]) -> Optional[float]:
    # Prefer summary.total_energy_kwh; fallback to last row of emissions.csv
    kwh = summary.get("total_energy_kwh")
    if isinstance(kwh, (int, float)) and pd.notna(kwh):
        return float(kwh)
    if emissions_df is not None and not emissions_df.empty:
        col = "energy_consumed"
        if col in emissions_df.columns:
            try:
                return float(emissions_df[col].iloc[-1])
            except Exception:
                pass
    return None

def is_online(summary: Dict[str, Any]) -> bool:
    return summary.get("tracker_mode") == "online"

def is_offline_kor(summary: Dict[str, Any]) -> bool:
    return summary.get("tracker_mode") == "offline" and summary.get("country_iso_code") == "KOR"

# ------------------------- UI: Load runs (preload + uploads) ---------------
with st.sidebar:
    st.subheader("Data")
    st.caption("Upload your run ZIPs or preload demos from the repository.")
    preload_demo = st.toggle("Preload bundled demo runs (sample_runs/)", value=True)
    uploads = st.file_uploader("Upload run ZIP(s)", type=["zip"], accept_multiple_files=True, label_visibility="visible")

runs: List[Tuple[str, Dict[str, Any], pd.DataFrame, pd.DataFrame]] = []

# Preload from repo
if preload_demo:
    for zp in list_repo_zip_paths():
        try:
            rd = extract_zip_path_to_tmp(zp)
            summary, epochs, samples, emissions_df = load_run_dir(rd)
            label = run_label(summary)
            runs.append((label, summary, epochs, samples))
            st.sidebar.success(f"Preloaded: sample_runs/{zp.name}")
        except Exception as e:
            st.sidebar.warning(f"Skipped {zp.name}: {e}")

# User uploads
if uploads:
    for up in uploads:
        try:
            rd = extract_zip_to_tmp(up)
            summary, epochs, samples, emissions_df = load_run_dir(rd)
            label = run_label(summary)
            runs.append((label, summary, epochs, samples))
            st.sidebar.success(f"Loaded: {up.name}")
        except Exception as e:
            st.sidebar.error(f"Failed {up.name}: {e}")

if not runs:
    st.info("No runs loaded yet. Use the sidebar to **upload ZIPs** or **enable preload**.")
    st.stop()

# ------------------------- Pick a baseline run -----------------------------
labels = [r[0] for r in runs]
baseline_idx = st.sidebar.selectbox("Choose a baseline run", options=list(range(len(labels))), format_func=lambda i: labels[i], index=0)
baseline_label, baseline_summary, baseline_epochs, baseline_samples = runs[baseline_idx]

# ------------------------- Tabs -------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Baseline", "Region & PUE", "Online vs Offline (Korea)", "Multi-run overlay"])

# ------------------------- Helpers: Plot styling ---------------------------
def _white_fig(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax

def _metric_row(summary: Dict[str, Any], epochs: pd.DataFrame):
    dur = summary.get("duration_s")
    kwh = summary.get("total_energy_kwh")
    kg  = summary.get("total_emissions_kg")
    val_acc = epochs["val_acc"].dropna().tail(1).iloc[0] * 100.0 if "val_acc" in epochs and epochs["val_acc"].notna().any() else None
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{float(dur):.1f} s" if dur else "—")
    col2.metric("Energy", f"{float(kwh):.4f} kWh" if kwh is not None else "—")
    col3.metric("Emissions", f"{float(kg):.4f} kg CO₂e" if kg is not None else "—")
    col4.metric("Final Val Acc", f"{val_acc:.2f} %" if val_acc is not None else "—")

# ------------------------- TAB 1: Baseline --------------------------------
with tab1:
    st.subheader("Baseline dashboard")
    st.caption(baseline_label)
    _metric_row(baseline_summary, baseline_epochs)

    # 1) Emissions per epoch (kg)
    if "emissions_kg" in baseline_epochs:
        fig, ax = _white_fig()
        ax.bar(baseline_epochs["epoch"], baseline_epochs["emissions_kg"].fillna(0.0))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("kg CO₂e")
        ax.set_title("Emissions per epoch")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # 2) Cumulative emissions (kg)
        fig, ax = _white_fig()
        cum = baseline_epochs["emissions_kg"].fillna(0.0).cumsum()
        ax.plot(baseline_epochs["epoch"], cum, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cumulative kg CO₂e")
        ax.set_title("Cumulative emissions")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # 3) Utilization over time (GPU if present else CPU)
    if not baseline_samples.empty and "timestamp" in baseline_samples.columns:
        t0 = baseline_samples["timestamp"].iloc[0]
        tmin = (baseline_samples["timestamp"] - t0) / 60.0
        fig, ax = _white_fig()
        if baseline_samples["gpu_util_pct"].notna().any():
            ax.plot(tmin, baseline_samples["gpu_util_pct"].fillna(0.0))
            ax.set_ylabel("GPU Utilization (%)")
            ax.set_title("GPU utilization over time")
        else:
            ax.plot(tmin, baseline_samples["cpu_util_pct"].fillna(0.0))
            ax.set_ylabel("CPU Utilization (%)")
            ax.set_title("CPU utilization over time")
        ax.set_xlabel("Minutes")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # 4) Accuracy vs emissions
    if "val_acc" in baseline_epochs and baseline_epochs["val_acc"].notna().any():
        fig, ax = _white_fig()
        ax.scatter(baseline_epochs["val_acc"] * 100.0, baseline_epochs["emissions_kg"].fillna(0.0))
        ax.set_xlabel("Validation accuracy (%)")
        ax.set_ylabel("kg CO₂e per epoch")
        ax.set_title("Trade-off: accuracy vs emissions")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

# ------------------------- TAB 2: Region & PUE (what-if) ------------------
with tab2:
    st.subheader("Region & PUE sensitivity (what-if)")

    # Need measured kWh
    # For best robustness, try to read emissions.csv if total_energy_kwh is None — not needed here because we only store summary in memory.
    measured = baseline_summary.get("total_energy_kwh")
    baseline_pue = float(baseline_summary.get("pue") or 1.2)

    if measured is None:
        st.warning("This baseline run is missing total_energy_kwh in summary.json; ensure CodeCarbon wrote energy_consumed.")
    else:
        regions = ["KOR", "MEX", "CAN", "FRA", "MNG"]
        pue_values = sorted({round(baseline_pue, 2), 1.2, 1.6})

        rows = []
        for r in regions:
            ef = float(EMISSION_FACTORS[r])
            for p in pue_values:
                emissions_kg = measured * p * ef
                rows.append({
                    "iso": r,
                    "region": REGION_NAME[r],
                    "pue": p,
                    "grid_factor_kg_per_kwh": ef,
                    "device_energy_kwh": measured,
                    "emissions_kg": emissions_kg,
                })
        df = pd.DataFrame(rows).sort_values(["emissions_kg", "region", "pue"]).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

        # Bar chart
        fig, ax = _white_fig(figsize=(9, 4))
        labels_bars = [f"{row['region']} | PUE={row['pue']}" for _, row in df.iterrows()]
        ax.bar(labels_bars, df["emissions_kg"])
        ax.set_xticklabels(labels_bars, rotation=45, ha="right")
        ax.set_ylabel("Emissions (kg CO₂e)")
        ax.set_title("Region & PUE sensitivity (fixed measured kWh)")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.caption(
            f"Measured device energy: **{measured:.4f} kWh** · Baseline PUE: **{baseline_pue:.2f}**"
        )

# ------------------------- TAB 3: Online vs Offline (Korea) ---------------
with tab3:
    st.subheader("Online vs Offline (Korea)")

    # Find a pair among loaded runs
    online_pairs = [(lbl, s, e, smp) for (lbl, s, e, smp) in runs if is_online(s)]
    kor_pairs    = [(lbl, s, e, smp) for (lbl, s, e, smp) in runs if is_offline_kor(s)]

    if not online_pairs or not kor_pairs:
        st.info("Load at least one ONLINE run and one OFFLINE-KOR run to view this comparison.")
    else:
        # pick the first of each (you can expand to let user pick)
        on_label, on_sum, on_epochs, _ = online_pairs[0]
        ko_label, ko_sum, ko_epochs, _ = kor_pairs[0]

        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"**ONLINE**: {on_label}")
            _metric_row(on_sum, on_epochs)
        with colB:
            st.markdown(f"**OFFLINE (Korea)**: {ko_label}")
            _metric_row(ko_sum, ko_epochs)

        # Overlay: cumulative emissions
        if "emissions_kg" in on_epochs and "emissions_kg" in ko_epochs:
            fig, ax = _white_fig(figsize=(8, 4))
            ax.plot(on_epochs["epoch"], on_epochs["emissions_kg"].fillna(0.0).cumsum(), marker="o", label="Online (auto)")
            ax.plot(ko_epochs["epoch"], ko_epochs["emissions_kg"].fillna(0.0).cumsum(), marker="o", label="Offline (KOR)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cumulative kg CO₂e")
            ax.set_title("Cumulative emissions — Online vs Offline (Korea)")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        # Overlay: per-epoch emissions
        if "emissions_kg" in on_epochs and "emissions_kg" in ko_epochs:
            fig, ax = _white_fig(figsize=(8, 4))
            ax.plot(on_epochs["epoch"], on_epochs["emissions_kg"].fillna(0.0), marker="o", label="Online (auto)")
            ax.plot(ko_epochs["epoch"], ko_epochs["emissions_kg"].fillna(0.0), marker="o", label="Offline (KOR)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("kg CO₂e per epoch")
            ax.set_title("Per-epoch emissions — Online vs Offline (Korea)")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

# ------------------------- TAB 4: Multi-run overlay -----------------------
with tab4:
    st.subheader("Multi-run overlay (cumulative emissions)")
    st.caption("Compare any number of loaded runs. Select which to include below.")

    # choose runs to plot
    chosen = st.multiselect("Select runs to overlay", options=list(range(len(labels))), default=list(range(len(labels))), format_func=lambda i: labels[i])
    if chosen:
        fig, ax = _white_fig(figsize=(9, 4))
        for i in chosen:
            lbl, s, e, smp = runs[i]
            if "emissions_kg" in e:
                ax.plot(e["epoch"], e["emissions_kg"].fillna(0.0).cumsum(), marker="o", label=lbl)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cumulative kg CO₂e")
        ax.set_title("Cumulative emissions overlay")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Small table of totals for quick comparison
        rows = []
        for i in chosen:
            lbl, s, e, _ = runs[i]
            total = float(e["emissions_kg"].fillna(0.0).sum()) if "emissions_kg" in e else None
            rows.append({"run": lbl, "total_kg_CO2e (sum of epochs)": total})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ------------------------- Footer -----------------------------------------
with st.expander("About this app"):
    st.markdown(
        """
        - **CodeCarbon** is used in your notebook to estimate energy and emissions during training.
        - This app expects each run ZIP to contain `summary.json`, `epochs.csv`, `samples.csv`, `emissions.csv`.
        - **Online mode** relies on CodeCarbon's geolocation / provider defaults.  
        - **Offline mode** uses an explicit `COUNTRY_ISO_CODE` (e.g., KOR) and optional `PUE`.
        - Region & PUE sensitivity is a **what-if** analysis that holds measured IT energy constant and varies grid factor & PUE.
        """
    )

