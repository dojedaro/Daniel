import io
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================
# App Constants & Theme
# =============================
APP_TITLE = "Carbon Footprint Calculator for AI Models — Daniel Ojeda Rosales"
REQUIRED_FILES = {"summary.json", "epochs.csv", "samples.csv", "emissions.csv"}
SAMPLE_RUNS_DIR = Path("sample_runs")

# Country display (emoji + color). Colors chosen for distinctness on light plots.
COUNTRY_META = {
    "KOR": {"name": "Korea", "flag": "🇰🇷", "color": "#1f77b4"},
    "MEX": {"name": "Mexico", "flag": "🇲🇽", "color": "#ff7f0e"},
    "CAN": {"name": "Canada", "flag": "🇨🇦", "color": "#2ca02c"},
    "FRA": {"name": "France", "flag": "🇫🇷", "color": "#d62728"},
    "MNG": {"name": "Mongolia", "flag": "🇲🇳", "color": "#9467bd"},
}

# Default kgCO2e per kWh factors (approximate; editable in sidebar). These act as fallbacks
# if CodeCarbon's internal mappings aren't available at runtime.
DEFAULT_COUNTRY_FACTORS = {
    "KOR": 0.45,
    "MEX": 0.40,
    "CAN": 0.12,
    "FRA": 0.06,
    "MNG": 0.80,
}

# =============================
# Utility Functions
# =============================

def _safe_json_load(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _find_run_root(extracted_dir: Path) -> Optional[Path]:
    """
    Robustly find the run folder containing REQUIRED_FILES under extracted_dir.
    Handles cases where the ZIP contains an extra parent folder or is nested.
    """
    # 1) Direct hit at this level
    entries = {p.name for p in extracted_dir.iterdir() if p.is_file()}
    if REQUIRED_FILES.issubset(entries):
        return extracted_dir

    # 2) Search one level down for a directory that contains required files
    for child in extracted_dir.iterdir():
        if child.is_dir():
            entries = {p.name for p in child.iterdir() if p.is_file()}
            if REQUIRED_FILES.issubset(entries):
                return child

    # 3) Recursive search just in case of deeper nesting
    for root, dirs, files in os.walk(extracted_dir):
        if REQUIRED_FILES.issubset(set(files)):
            return Path(root)

    return None


def _parse_emissions_kwh(emissions_csv: Path) -> float:
    """Try common columns to extract measured kWh from CodeCarbon logs."""
    df = pd.read_csv(emissions_csv)
    cols = list(df.columns)
    candidates = [
        "energy_consumed",  # kWh in many CodeCarbon versions
        "energy_kwh",
        "energy",  # sometimes in Wh; we will detect by magnitude
    ]
    for c in candidates:
        if c in cols:
            kwh = df[c].sum()
            # Heuristic: if suspiciously large, it may be Wh — convert to kWh
            if kwh > 1e4:
                kwh = kwh / 1000.0
            return float(kwh)
    return 0.0


def _read_epochs(epochs_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(epochs_csv)
    # Ensure epoch column
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
    # Normalize potential per-epoch kWh column names
    if "kwh" not in df.columns:
        for cand in ["energy_kwh", "epoch_kwh", "energy", "energy_consumed_kwh"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "kwh"})
                break
    # Create placeholders to be possibly filled later
    if "emissions_kg" not in df.columns:
        df["emissions_kg"] = np.nan
    if "cumulative_emissions_kg" not in df.columns:
        df["cumulative_emissions_kg"] = np.nan
    return df


def _read_samples(samples_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(samples_csv)
    # Expect columns like: timestamp, cpu_percent, gpu_util, mem_gb, gpu_mem_gb
    # We'll standardize if we can.
    rename_map = {}
    for cand in ["cpu", "cpu_percent", "cpu_util", "cpu_usage"]:
        if cand in df.columns:
            rename_map[cand] = "cpu_percent"
            break
    for cand in ["gpu", "gpu_util", "gpu_percent", "gpu_usage"]:
        if cand in df.columns:
            rename_map[cand] = "gpu_util"
            break
    for cand in ["mem_gb", "memory_gb", "ram_gb"]:
        if cand in df.columns:
            rename_map[cand] = "mem_gb"
            break
    for cand in ["gpu_mem_gb", "vram_gb"]:
        if cand in df.columns:
            rename_map[cand] = "gpu_mem_gb"
            break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def read_run_folder(run_root: Path) -> Dict[str, object]:
    summary = _safe_json_load(run_root / "summary.json")
    epochs = _read_epochs(run_root / "epochs.csv")
    samples = _read_samples(run_root / "samples.csv")

    emissions_kwh = _parse_emissions_kwh(run_root / "emissions.csv")

    # Try to derive total kgCO2e from emissions.csv if present
    kg_total = None
    try:
        df_e = pd.read_csv(run_root / "emissions.csv")
        for c in ["emissions", "emissions_kg", "co2e", "carbon_emissions"]:
            if c in df_e.columns:
                kg_total = float(df_e[c].sum())
                break
    except Exception:
        kg_total = None

    # If we have per-epoch kWh and a total kg, derive kg/kWh factor and fill per-epoch emissions
    kg_per_kwh = None
    if emissions_kwh and kg_total and emissions_kwh > 0:
        kg_per_kwh = kg_total / emissions_kwh

    # Fill emissions_kg and cumulative if missing
    if epochs["emissions_kg"].isna().all():
        if "kwh" in epochs.columns and epochs["kwh"].notna().any() and kg_per_kwh:
            epochs["emissions_kg"] = epochs["kwh"].fillna(0) * kg_per_kwh
        elif kg_total is not None and len(epochs) > 0:
            # Even split fallback when we only know totals
            per_epoch = kg_total / len(epochs)
            epochs["emissions_kg"] = per_epoch

    if epochs["cumulative_emissions_kg"].isna().all() and epochs["emissions_kg"].notna().any():
        epochs["cumulative_emissions_kg"] = epochs["emissions_kg"].fillna(0).cumsum()

    return {
        "root": run_root,
        "summary": summary,
        "epochs": epochs,
        "samples": samples,
        "measured_kwh": emissions_kwh,
        "measured_kg": kg_total,
        "kg_per_kwh": kg_per_kwh,
    }


def extract_zip_to_tmp(file_like: io.BytesIO, label: str) -> Optional[Path]:
    import zipfile

    tmpdir = Path(tempfile.mkdtemp(prefix=f"runzip_{label}_"))
    try:
        with zipfile.ZipFile(file_like) as zf:
            zf.extractall(tmpdir)
        run_root = _find_run_root(tmpdir)
        if run_root is None:
            st.error(f"Could not locate a run folder with required files in '{label}'.")
            return None
        return run_root
    except zipfile.BadZipFile:
        st.error(f"The file '{label}' is not a valid ZIP.")
        return None


def list_repo_zip_paths() -> List[Path]:
    if not SAMPLE_RUNS_DIR.exists():
        return []
    return sorted(SAMPLE_RUNS_DIR.glob("*.zip"))


# =============================
# Plot Helpers (Plotly, white background)
# =============================

def _apply_plot_defaults(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    return fig


def plot_epoch_emissions(df: pd.DataFrame) -> go.Figure:
    y = df["emissions_kg"] if "emissions_kg" in df.columns else None
    if y is None or y.isna().all():
        # If epoch-level emissions aren't available, show a placeholder
        fig = go.Figure()
        fig.add_annotation(text="Per-epoch emissions not found in epochs.csv",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_plot_defaults(fig)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["epoch"], y=y, name="Emissions per epoch (kgCO₂e)"))
    fig.update_yaxes(title_text="kgCO₂e")
    fig.update_xaxes(title_text="Epoch")
    return _apply_plot_defaults(fig)


def plot_cumulative_emissions(df: pd.DataFrame) -> go.Figure:
    y = df["cumulative_emissions_kg"] if "cumulative_emissions_kg" in df.columns else None
    if y is None or y.isna().all():
        fig = go.Figure()
        fig.add_annotation(text="Cumulative emissions not found; provide emissions_kg in epochs.csv",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_plot_defaults(fig)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["epoch"], y=y, mode="lines+markers",
                             name="Cumulative emissions (kgCO₂e)"))
    fig.update_yaxes(title_text="kgCO₂e")
    fig.update_xaxes(title_text="Epoch")
    return _apply_plot_defaults(fig)


def plot_utilization(samples: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "cpu_percent" in samples.columns:
        fig.add_trace(go.Scatter(x=np.arange(len(samples)), y=samples["cpu_percent"],
                                 mode="lines", name="CPU %"))
    if "gpu_util" in samples.columns:
        fig.add_trace(go.Scatter(x=np.arange(len(samples)), y=samples["gpu_util"],
                                 mode="lines", name="GPU %"))
    fig.update_yaxes(title_text="Utilization (%)", range=[0, 100])
    fig.update_xaxes(title_text="Sample index (1s)")
    return _apply_plot_defaults(fig)


def plot_accuracy_vs_emissions(df: pd.DataFrame) -> go.Figure:
    # Look for common accuracy/metric columns
    acc_col = None
    for cand in ["accuracy", "acc", "val_accuracy", "top1_acc"]:
        if cand in df.columns:
            acc_col = cand
            break
    emis_col = None
    for cand in ["cumulative_emissions_kg", "emissions_kg"]:
        if cand in df.columns and df[cand].notna().any():
            emis_col = cand
            break
    fig = go.Figure()
    if acc_col and emis_col:
        fig.add_trace(go.Scatter(x=df[emis_col], y=df[acc_col], mode="markers+lines",
                                 name=f"{acc_col} vs {emis_col}"))
        fig.update_xaxes(title_text="Emissions (kgCO₂e)")
        fig.update_yaxes(title_text=acc_col)
    else:
        fig.add_annotation(text="Couldn't find accuracy/emissions columns in epochs.csv",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return _apply_plot_defaults(fig)


def plot_country_bars(measured_kwh: float, factors: Dict[str, float], pue: float) -> go.Figure:
    fig = go.Figure()
    for iso, meta in COUNTRY_META.items():
        name = f"{meta['flag']} {meta['name']}"
        color = meta["color"]
        kg = measured_kwh * pue * factors.get(iso, DEFAULT_COUNTRY_FACTORS.get(iso, 0.4))
        fig.add_trace(go.Bar(x=[name], y=[kg], name=name, marker_color=color))
    fig.update_yaxes(title_text="kgCO₂e")
    fig.update_xaxes(title_text="Country (what‑if)")
    return _apply_plot_defaults(fig, height=420)


def plot_multi_run_overlay(runs: Dict[str, Dict[str, object]]) -> go.Figure:
    fig = go.Figure()
    for label, data in runs.items():
        df = data["epochs"].copy()
        if "cumulative_emissions_kg" in df.columns and df["cumulative_emissions_kg"].notna().any():
            fig.add_trace(go.Scatter(x=df["epoch"], y=df["cumulative_emissions_kg"],
                                     mode="lines+markers", name=label))
        else:
            # Graceful message per missing series
            fig.add_annotation(text=f"No cumulative emissions for {label}",
                               xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False)
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Cumulative kgCO₂e")
    return _apply_plot_defaults(fig)


# =============================
# Sidebar Controls & Explanations
# =============================

def sidebar_explanations():
    st.sidebar.header("ℹ️ How to read the plots")
    st.sidebar.markdown(
        """
        **Baseline**
        - *Emissions per epoch*: bars show per‑epoch kgCO₂e (if available).
        - *Cumulative emissions*: line grows as training progresses.
        - *Utilization over time*: CPU/GPU percentages sampled each second.
        - *Accuracy vs emissions*: how performance improves as emissions accumulate.

        **Region & PUE**
        - Uses your **measured kWh** from a single run.
        - Applies **PUE** (data center overhead) and **country factors** (kgCO₂e/kWh).
        - Bars compare what emissions *would have been* in each country.

        **Online vs Offline (Korea)**
        - Select one **online** run and one **offline‑KOR** run.
        - Overlays cumulative emissions to compare geolocation vs fixed‑factor settings.

        **Multi‑run overlay**
        - Compare cumulative emissions curves across any number of runs.
        """
    )

    st.sidebar.header("⚙️ Factors (editable)")
    st.sidebar.caption(
        "Country factors are default kgCO₂e per kWh. Adjust if you have newer data."
    )
    factors = {}
    for iso, default in DEFAULT_COUNTRY_FACTORS.items():
        meta = COUNTRY_META[iso]
        factors[iso] = st.sidebar.number_input(
            f"{meta['flag']} {meta['name']} (kgCO₂e/kWh)",
            min_value=0.0, max_value=2.0, value=float(default), step=0.01,
        )

    pue = st.sidebar.number_input(
        "PUE (Power Usage Effectiveness)", min_value=1.0, max_value=3.0, value=1.2, step=0.05
    )
    return factors, pue


# =============================
# Streamlit App
# =============================

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")

    st.title(APP_TITLE)
    st.caption("Streamlit dark shell · charts on white for visibility · flags + distinct colors")

    # Sidebar: explanations + editable factors
    factors, pue = sidebar_explanations()

    # Preload Toggle
    st.subheader("Data Sources")
    c1, c2 = st.columns([1, 2])
    with c1:
        do_preload = st.checkbox("Preload bundled demo runs (sample_runs/)", value=False)
    preloaded_runs: Dict[str, Dict[str, object]] = {}

    if do_preload:
        zips = list_repo_zip_paths()
        if not zips:
            st.warning("No ZIPs found in ./sample_runs — check your deployment branch & files.")
        for z in zips:
            try:
                with open(z, "rb") as f:
                    data = io.BytesIO(f.read())
                run_root = extract_zip_to_tmp(data, label=z.name)
                if run_root is None:
                    continue
                run_data = read_run_folder(run_root)
                label = f"sample_runs/{z.name}"
                preloaded_runs[label] = run_data
                st.success(f"Preloaded: {label}")
            except Exception as e:
                st.error(f"Error preloading {z.name}: {e}")

    # Uploads
    uploaded_runs: Dict[str, Dict[str, object]] = {}
    with c2:
        up = st.file_uploader(
            "Upload one or more run ZIPs (each contains summary.json, epochs.csv, samples.csv, emissions.csv)",
            type=["zip"], accept_multiple_files=True,
        )
        if up:
            for f in up:
                run_root = extract_zip_to_tmp(f.getvalue() if hasattr(f, "getvalue") else f.read(), label=f.name)
                if run_root is None:
                    continue
                try:
                    run_data = read_run_folder(run_root)
                    uploaded_runs[f"upload/{f.name}"] = run_data
                    st.success(f"Loaded: upload/{f.name}")
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")

    # Merge all available runs
    all_runs = {**preloaded_runs, **uploaded_runs}
    if not all_runs:
        st.info("No runs loaded yet. Preload from sample_runs/ or upload ZIPs to begin.")
        st.stop()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Baseline",
        "Region & PUE (what‑if)",
        "Online vs Offline (Korea)",
        "Multi‑run overlay",
    ])

    # =============================
    # Baseline Tab
    # =============================
    with tab1:
        st.markdown("### Baseline: explore a single run")
        labels = list(all_runs.keys())
        sel = st.selectbox("Select a run", labels)
        data = all_runs[sel]
        summary = data["summary"]
        measured_kwh = data["measured_kwh"]
        epochs = data["epochs"]
        samples = data["samples"]

        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Measured energy (kWh)", f"{measured_kwh:.4f}")
        with cB:
            mode = summary.get("mode", summary.get("carbon_mode", "n/a")).upper() if isinstance(summary, dict) else "n/a"
            st.metric("Mode", mode)
        with cC:
            region = summary.get("region", summary.get("country", "n/a")) if isinstance(summary, dict) else "n/a"
            st.metric("Region", str(region))

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_epoch_emissions(epochs), use_container_width=True)
        with c2:
            st.plotly_chart(plot_cumulative_emissions(epochs), use_container_width=True)

        st.plotly_chart(plot_utilization(samples), use_container_width=True)
        st.plotly_chart(plot_accuracy_vs_emissions(epochs), use_container_width=True)

    # =============================
    # Region & PUE What‑If Tab
    # =============================
    with tab2:
        st.markdown("### Region & PUE what‑if: reuse measured kWh, vary location/PUE")
        labels = list(all_runs.keys())
        sel = st.selectbox("Select a base run (for measured kWh)", labels, key="whatif_select")
        measured_kwh = all_runs[sel]["measured_kwh"]
        if measured_kwh <= 0:
            st.error("Measured kWh couldn't be derived from emissions.csv. Check the file or columns.")
        st.plotly_chart(plot_country_bars(measured_kwh, factors, pue), use_container_width=True)
        st.caption("Bars show kgCO₂e = measured_kWh × PUE × country_factor. Flags and colors distinguish countries.")

    # =============================
    # Online vs Offline (Korea) Tab
    # =============================
    with tab3:
        st.markdown("### Online vs Offline (Korea): overlay cumulative emissions")
        labels = list(all_runs.keys())
        c1, c2 = st.columns(2)
        with c1:
            run_online = st.selectbox("Select ONLINE run", labels, key="ovoo_online")
        with c2:
            run_offkor = st.selectbox("Select OFFLINE‑KOR run", labels, key="ovoo_offkor")

        comparison = {}
        if run_online:
            comparison[f"Online — {run_online}"] = all_runs[run_online]
        if run_offkor and run_offkor != run_online:
            comparison[f"Offline‑KOR — {run_offkor}"] = all_runs[run_offkor]
        if len(comparison) < 2:
            st.info("Pick two different runs (one ONLINE, one OFFLINE‑KOR) to see the overlay.")
        st.plotly_chart(plot_multi_run_overlay(comparison), use_container_width=True)

    # =============================
    # Multi‑run Overlay Tab
    # =============================
    with tab4:
        st.markdown("### Multi‑run overlay: compare cumulative curves across runs")
        labels = list(all_runs.keys())
        selected = st.multiselect("Select runs to compare", labels, default=labels[: min(3, len(labels))])
        subset = {k: all_runs[k] for k in selected}
        if not subset:
            st.info("Select at least one run.")
        st.plotly_chart(plot_multi_run_overlay(subset), use_container_width=True)


if __name__ == "__main__":
    main()

