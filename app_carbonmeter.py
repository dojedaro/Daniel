import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

# Country display (emoji + color)
COUNTRY_META = {
    "KOR": {"name": "Korea", "flag": "🇰🇷", "color": "#1f77b4"},
    "MEX": {"name": "Mexico", "flag": "🇲🇽", "color": "#ff7f0e"},
    "CAN": {"name": "Canada", "flag": "🇨🇦", "color": "#2ca02c"},
    "FRA": {"name": "France", "flag": "🇫🇷", "color": "#d62728"},
    "MNG": {"name": "Mongolia", "flag": "🇲🇳", "color": "#9467bd"},
}

# Default kgCO₂e/kWh factors (editable)
DEFAULT_COUNTRY_FACTORS = {"KOR": 0.45, "MEX": 0.40, "CAN": 0.12, "FRA": 0.06, "MNG": 0.80}


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
    Handles Zips with parent folders or deeper nesting.
    """
    # 1) Direct hit at this level
    entries = {p.name for p in extracted_dir.iterdir() if p.is_file()}
    if REQUIRED_FILES.issubset(entries):
        return extracted_dir

    # 2) Search one level down
    for child in extracted_dir.iterdir():
        if child.is_dir():
            entries = {p.name for p in child.iterdir() if p.is_file()}
            if REQUIRED_FILES.issubset(entries):
                return child

    # 3) Recursive search
    for root, _, files in os.walk(extracted_dir):
        if REQUIRED_FILES.issubset(set(files)):
            return Path(root)

    return None


def _parse_emissions_kwh(emissions_csv: Path) -> float:
    """
    Extract measured kWh from CodeCarbon logs.
    Prefer the *last* cumulative value if available; avoid summing intervals.
    """
    df = pd.read_csv(emissions_csv)
    if df.empty:
        return 0.0

    for cumulative_col in ["energy_consumed", "energy_kwh"]:
        if cumulative_col in df.columns:
            v = df[cumulative_col].dropna().iloc[-1]
            return float(v)

    if "energy" in df.columns:
        v = float(df["energy"].dropna().iloc[-1])
        if v > 1e4:   # likely Wh
            return v / 1000.0
        return v

    return 0.0


def _read_epochs(epochs_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(epochs_csv)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))

    if "kwh" not in df.columns:
        for cand in ["energy_kwh", "epoch_kwh", "energy", "energy_consumed_kwh"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "kwh"})
                break

    if "emissions_kg" not in df.columns:
        df["emissions_kg"] = np.nan
    if "cumulative_emissions_kg" not in df.columns:
        df["cumulative_emissions_kg"] = np.nan
    return df


def _read_samples(samples_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(samples_csv)

    # Normalize columns from UsageSampler → app expectations
    rename_map = {}

    # CPU %
    if "cpu_util_pct" in df.columns:
        rename_map["cpu_util_pct"] = "cpu_percent"
    else:
        for cand in ["cpu", "cpu_percent", "cpu_util", "cpu_usage"]:
            if cand in df.columns:
                rename_map[cand] = "cpu_percent"
                break

    # GPU %
    if "gpu_util_pct" in df.columns:
        rename_map["gpu_util_pct"] = "gpu_util"
    else:
        for cand in ["gpu", "gpu_util", "gpu_percent", "gpu_usage"]:
            if cand in df.columns:
                rename_map[cand] = "gpu_util"
                break

    # System RAM (GB)
    if "cpu_mem_gb_used" in df.columns:
        rename_map["cpu_mem_gb_used"] = "mem_gb"
    else:
        for cand in ["mem_gb", "memory_gb", "ram_gb"]:
            if cand in df.columns:
                rename_map[cand] = "mem_gb"
                break

    # GPU VRAM (GB)
    if "gpu_mem_gb_used" in df.columns:
        rename_map["gpu_mem_gb_used"] = "gpu_mem_gb"
    elif "vram_gb" in df.columns:
        rename_map["vram_gb"] = "gpu_mem_gb"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def read_run_folder(run_root: Path) -> Dict[str, object]:
    summary = _safe_json_load(run_root / "summary.json")
    epochs = _read_epochs(run_root / "epochs.csv")
    samples = _read_samples(run_root / "samples.csv")

    emissions_kwh = _parse_emissions_kwh(run_root / "emissions.csv")

    # Try to derive total kgCO₂e from emissions.csv if present
    kg_total = None
    try:
        df_e = pd.read_csv(run_root / "emissions.csv")
        for c in ["emissions", "emissions_kg", "co2e", "carbon_emissions"]:
            if c in df_e.columns:
                kg_total = float(df_e[c].dropna().iloc[-1]) if df_e[c].notna().any() else None
                # some versions log cumulative; if per-interval, sum as fallback
                if kg_total is None or kg_total == 0.0:
                    kg_total = float(df_e[c].sum())
                break
    except Exception:
        kg_total = None

    # If we have per-epoch kWh and a total kg, derive kg/kWh factor
    kg_per_kwh = None
    if emissions_kwh and kg_total and emissions_kwh > 0:
        kg_per_kwh = kg_total / emissions_kwh

    # Fill per-epoch emissions if missing
    if epochs["emissions_kg"].isna().all():
        if "kwh" in epochs.columns and epochs["kwh"].notna().any() and kg_per_kwh:
            epochs["emissions_kg"] = epochs["kwh"].fillna(0) * kg_per_kwh
        elif kg_total is not None and len(epochs) > 0:
            per_epoch = kg_total / len(epochs)
            epochs["emissions_kg"] = per_epoch

    # Fill cumulative if missing
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


def extract_zip_to_tmp(file_like, label: str) -> Optional[Path]:
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
# Plot Helpers (Plotly)
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
    fig = go.Figure()
    if y is None or y.isna().all():
        fig.add_annotation(text="Per-epoch emissions not found in epochs.csv",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_plot_defaults(fig)
    fig.add_trace(go.Bar(x=df["epoch"], y=y, name="Emissions per epoch (kgCO₂e)"))
    fig.update_yaxes(title_text="kgCO₂e")
    fig.update_xaxes(title_text="Epoch")
    return _apply_plot_defaults(fig)


def plot_cumulative_emissions(df: pd.DataFrame) -> go.Figure:
    y = df["cumulative_emissions_kg"] if "cumulative_emissions_kg" in df.columns else None
    fig = go.Figure()
    if y is None or y.isna().all():
        fig.add_annotation(text="Cumulative emissions not found; provide emissions_kg in epochs.csv",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_plot_defaults(fig)
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
    acc_col = None
    for cand in ["val_acc", "accuracy", "acc", "val_accuracy", "top1_acc"]:
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
    fig.update_xaxes(title_text="Country (what-if)")
    return _apply_plot_defaults(fig, height=420)


def plot_multi_run_overlay(runs: Dict[str, Dict[str, object]]) -> go.Figure:
    fig = go.Figure()
    added_any = False
    for label, data in runs.items():
        df = data["epochs"].copy()
        if "cumulative_emissions_kg" in df.columns and df["cumulative_emissions_kg"].notna().any():
            fig.add_trace(go.Scatter(x=df["epoch"], y=df["cumulative_emissions_kg"],
                                     mode="lines+markers", name=label))
            added_any = True
    if not added_any:
        fig.add_annotation(text="No cumulative emissions series to show yet.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
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
        - *Emissions per epoch*: bars show per-epoch kgCO₂e (if available).
        - *Cumulative emissions*: line grows as training progresses.
        - *Utilization over time*: CPU/GPU percentage each second.
        - *Accuracy vs emissions*: performance as emissions accumulate.

        **Region & PUE**
        - Uses your **measured kWh** from a run.
        - Formula: **kgCO₂e = measured_kWh × PUE × country_factor**.
        - Adjust factors in the sidebar.

        **Online vs Offline (Korea)**
        - Choose one **ONLINE** and one **OFFLINE-KOR** run to overlay curves.

        **Multi-run**
        - Overlay cumulative emissions curves for any runs you load.
        """
    )

    st.sidebar.header("⚙️ Factors (editable)")
    st.sidebar.caption("Country factors are kgCO₂e per kWh. Adjust if you have newer data.")
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

    # Data Sources
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
            "Upload one or more run ZIPs (contains summary.json, epochs.csv, samples.csv, emissions.csv)",
            type=["zip"], accept_multiple_files=True,
        )
        if up:
            for f in up:
                try:
                    data = io.BytesIO(f.getvalue())   # ensure file-like
                    run_root = extract_zip_to_tmp(data, label=f.name)
                    if run_root is None:
                        continue
                    run_data = read_run_folder(run_root)
                    uploaded_runs[f"upload/{f.name}"] = run_data
                    st.success(f"Loaded: upload/{f.name}")
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")

    # Merge runs
    all_runs = {**preloaded_runs, **uploaded_runs}
    if not all_runs:
        st.info("No runs loaded yet. Preload from sample_runs/ or upload ZIPs to begin.")
        st.stop()

    # Aggregate table for comparisons
    def aggregate_runs(runs_dict: Dict[str, Dict[str, object]]) -> pd.DataFrame:
        rows = []
        for label, data in runs_dict.items():
            summary = data["summary"] or {}
            measured_kwh = data["measured_kwh"]
            measured_kg = data["measured_kg"]
            epochs_df = data["epochs"]
            final_acc = None
            if not epochs_df.empty and "val_acc" in epochs_df.columns and epochs_df["val_acc"].notna().any():
                final_acc = float(epochs_df["val_acc"].dropna().iloc[-1])
            tracker_mode = (summary.get("tracker_mode") or "n/a").upper()
            region = summary.get("country_iso_code") or "ONLINE"
            pue_val = summary.get("pue", None)
            epochs_ct = summary.get("epochs") or (len(epochs_df) if not epochs_df.empty else None)
            kwh_per_epoch = (measured_kwh / epochs_ct) if (measured_kwh and epochs_ct) else None
            kg_per_epoch = (measured_kg / epochs_ct) if (measured_kg and epochs_ct) else None
            rows.append({
                "Run": label,
                "Mode": tracker_mode,
                "Region": region,
                "PUE": pue_val,
                "Epochs": epochs_ct,
                "Duration (s)": summary.get("duration_s"),
                "kWh": measured_kwh,
                "kg CO₂e": measured_kg,
                "Val Acc": final_acc,
                "kWh / epoch": kwh_per_epoch,
                "kg / epoch": kg_per_epoch,
            })
        return pd.DataFrame(rows)

    agg_df = aggregate_runs(all_runs)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Baseline",
        "Region & PUE (what-if)",
        "Online vs Offline (Korea)",
        "Multi-run overlay",
        "Savings vs Baseline",
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
        measured_kwh = float(data["measured_kwh"] or 0.0)
        epochs = data["epochs"]
        samples = data["samples"]

        cA, cB, cC, cD = st.columns(4)
        with cA:
            st.metric("Measured energy (kWh)", f"{measured_kwh:.6f}")
        tracker_mode = (summary.get("tracker_mode") or "n/a").upper()
        with cB:
            st.metric("Mode", tracker_mode)
        country = summary.get("country_iso_code")
        region_label = country if country else "ONLINE"
        with cC:
            st.metric("Region", region_label)
        pue_val = summary.get("pue", None)
        with cD:
            st.metric("PUE", f"{pue_val:.2f}" if pue_val is not None else "n/a")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_epoch_emissions(epochs), use_container_width=True)
        with c2:
            st.plotly_chart(plot_cumulative_emissions(epochs), use_container_width=True)

        st.plotly_chart(plot_utilization(samples), use_container_width=True)
        st.plotly_chart(plot_accuracy_vs_emissions(epochs), use_container_width=True)

    # =============================
    # Region & PUE What-If Tab
    # =============================
    with tab2:
        st.markdown("### Region & PUE what-if: reuse measured kWh, vary location/PUE")
        labels = list(all_runs.keys())
        sel = st.selectbox("Select a base run (for measured kWh)", labels, key="whatif_select")
        measured_kwh = float(all_runs[sel]["measured_kwh"] or 0.0)
        if measured_kwh <= 0:
            st.error("Measured kWh couldn't be derived from emissions.csv. Check the file/columns.")
        st.plotly_chart(plot_country_bars(measured_kwh, factors, pue), use_container_width=True)
        st.caption("Formula: kgCO₂e = measured_kWh × PUE × country_factor. Flags and colors distinguish countries.")

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
            run_offkor = st.selectbox("Select OFFLINE-KOR run", labels, key="ovoo_offkor")

        comparison = {}
        if run_online:
            comparison[f"Online — {run_online}"] = all_runs[run_online]
        if run_offkor and run_offkor != run_online:
            comparison[f"Offline-KOR — {run_offkor}"] = all_runs[run_offkor]
        if len(comparison) < 2:
            st.info("Pick two different runs (one ONLINE, one OFFLINE-KOR) to see the overlay.")
        st.plotly_chart(plot_multi_run_overlay(comparison), use_container_width=True)

    # =============================
    # Multi-run Overlay Tab
    # =============================
    with tab4:
        st.markdown("### Multi-run overlay: compare cumulative curves across runs")
        labels = list(all_runs.keys())
        selected = st.multiselect("Select runs to compare", labels, default=labels[: min(3, len(labels))])
        subset = {k: all_runs[k] for k in selected}
        if not subset:
            st.info("Select at least one run.")
        st.plotly_chart(plot_multi_run_overlay(subset), use_container_width=True)

    # =============================
    # Savings vs Baseline Tab
    # =============================
    with tab5:
        st.markdown("### Savings vs Baseline")
        if agg_df.empty:
            st.info("Load at least one run to compare.")
        else:
            st.dataframe(agg_df)

            baseline_name = st.selectbox("Choose a baseline run", agg_df["Run"].tolist(), index=0, key="baseline_sel")
            base_row = agg_df[agg_df["Run"] == baseline_name].iloc[0]

            def pct_change(v, base):
                if base in [None, 0] or pd.isna(base) or v is None or pd.isna(v):
                    return np.nan
                return 100.0 * (v - base) / base

            savings = pd.DataFrame({
                "Run": agg_df["Run"],
                "%Δ Duration": agg_df["Duration (s)"].apply(lambda v: pct_change(v, base_row["Duration (s)"])),
                "%Δ kWh": agg_df["kWh"].apply(lambda v: pct_change(v, base_row["kWh"])),
                "%Δ kg CO₂e": agg_df["kg CO₂e"].apply(lambda v: pct_change(v, base_row["kg CO₂e"])),
                "%Δ kWh/epoch": agg_df["kWh / epoch"].apply(lambda v: pct_change(v, base_row["kWh / epoch"])),
                "%Δ kg/epoch": agg_df["kg / epoch"].apply(lambda v: pct_change(v, base_row["kg / epoch"])),
            })

            st.subheader("✅ Percent Savings vs Baseline (negative is better)")
            st.dataframe(
                savings.style.format({
                    "%Δ Duration": "{:+.1f}%",
                    "%Δ kWh": "{:+.1f}%",
                    "%Δ kg CO₂e": "{:+.1f}%",
                    "%Δ kWh/epoch": "{:+.1f}%",
                    "%Δ kg/epoch": "{:+.1f}%"
                })
            )

            # Quick bars for kWh and kg CO2e
            st.markdown("#### Energy & Emissions by Run")
            fig1 = go.Figure([go.Bar(x=agg_df["Run"], y=agg_df["kWh"], name="kWh")])
            fig1.update_yaxes(title_text="kWh")
            st.plotly_chart(_apply_plot_defaults(fig1, height=340), use_container_width=True)

            fig2 = go.Figure([go.Bar(x=agg_df["Run"], y=agg_df["kg CO₂e"], name="kg CO₂e")])
            fig2.update_yaxes(title_text="kg CO₂e")
            st.plotly_chart(_apply_plot_defaults(fig2, height=340), use_container_width=True)

            # Download aggregated CSV
            csv_buf = io.StringIO()
            agg_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download aggregated CSV",
                data=csv_buf.getvalue(),
                file_name=f"carboncalc_runs_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
