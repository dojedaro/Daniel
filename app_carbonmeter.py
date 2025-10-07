# Carbon Footprint Calculator for AI Models — Daniel Ojeda Rosales

import json, io, os, zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# --- quick debug of paths & zips (put near the top, after imports) ---
from pathlib import Path
import streamlit as st

root = Path(__file__).resolve().parent
sr1 = root / "sample_runs"
sr2 = Path.cwd() / "sample_runs"
found = [p for p in (sr1, sr2) if p.exists()]
zips = []
for d in found:
    zips += list(d.glob("*.zip"))

st.caption(f"🛠 repo dir: {root}")
st.caption(f"🛠 sample_runs dirs found: {', '.join(map(str,found)) or 'none'}")
st.caption(f"🛠 zips: {', '.join(p.name for p in zips) or 'none'}")

APP_TITLE = "Carbon Footprint Calculator for AI Models — Daniel Ojeda Rosales"

st.set_page_config(page_title=APP_TITLE, page_icon="🌿", layout="wide")

# Dark shell; figures will be white for readability
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.reportview-container, .main, .stApp { background: #0f1116 !important; color: #e2e2e2; }
.card { background: #161a22; border: 1px solid #2a2f3a; padding: 1rem; border-radius: 14px; }
hr { border: none; border-top: 1px solid #2a2f3a; margin: 0.5rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

st.title(APP_TITLE)
st.caption("Upload one or more *run ZIPs* (each containing summary.json, epochs.csv, samples.csv, emissions.csv).")

COUNTRY_COLOR = {"KOR":"#1f77b4","CAN":"#2ca02c","MEX":"#ff7f0e","MNG":"#9467bd"}

def fig_white(size=(7.5,4)):
    return plt.figure(figsize=size, facecolor="white")


# --- Robust ZIP extraction: find the true run root even if nested deeply ---
import io, zipfile
from pathlib import Path

REQUIRED_FILES = {"summary.json", "epochs.csv", "samples.csv", "emissions.csv"}

def _find_run_root(extracted_dir: Path) -> Path:
    # 1) If the files are right at this level
    names_here = {p.name for p in extracted_dir.iterdir() if p.is_file()}
    if REQUIRED_FILES.issubset(names_here):
        return extracted_dir
    # 2) Otherwise, search all subfolders for a directory containing all files
    for sj in extracted_dir.rglob("summary.json"):
        candidate = sj.parent
        if all((candidate / f).exists() for f in REQUIRED_FILES):
            return candidate
    raise FileNotFoundError(
        "Missing one or more required files (summary.json, epochs.csv, samples.csv, emissions.csv) in the uploaded ZIP."
    )

def extract_zip_to_tmp(uploaded_file) -> Path:
    out_dir = Path("/tmp") / f"run_{Path(uploaded_file.name).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    zbytes = io.BytesIO(uploaded_file.getvalue())
    with zipfile.ZipFile(zbytes, "r") as zf:
        zf.extractall(out_dir)
    return _find_run_root(out_dir)

def extract_zip_path_to_tmp(zip_path: Path) -> Path:
    out_dir = Path("/tmp") / f"run_{zip_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(out_dir)
    return _find_run_root(out_dir)



st.caption("Upload one or more *run ZIPs* OR let the app preload any demo ZIPs found under `sample_runs/` in this repo.")

preload_demo = st.toggle("Preload bundled demo runs (sample_runs/)", value=True)

uploads = st.file_uploader("Upload run ZIP(s)", type=["zip"], accept_multiple_files=True)

runs = []

# 1) Preload from repo/sample_runs (if toggle on)
if preload_demo:
    repo_zips = list_repo_zip_paths()
    for zp in repo_zips:
        try:
            rd = extract_zip_path_to_tmp(zp)
            summary, epochs, samples, emissions = load_run_dir(rd)
            runs.append((rd.name, summary, epochs, samples))
            st.success(f"Preloaded: sample_runs/{zp.name}")
        except Exception as e:
            st.warning(f"Skipped sample_runs/{zp.name}: {e}")

# 2) Also load any user uploads
if uploads:
    for up in uploads:
        try:
            rd = extract_zip_to_tmp(up)
            summary, epochs, samples, emissions = load_run_dir(rd)
            runs.append((rd.name, summary, epochs, samples))
            st.success(f"Loaded upload: {up.name}")
        except Exception as e:
            st.error(f"Failed to load {up.name}: {e}")

if not runs:
    st.info("No runs loaded yet. Upload ZIP(s) above or enable the preload toggle.")
    st.stop()


# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Baseline (single run)", "Online vs Offline (Korea)", "Region & PUE (what-if)", "Multi-run overlay"])

# -------- Baseline (uses first run) --------
with tab1:
    name, summary, epochs, samples = runs[0]
    st.subheader(f"Baseline — {name}")
    mode = summary.get("tracker_mode", "online").upper()
    iso  = summary.get("country_iso_code") or "auto"
    pue  = summary.get("pue", None)
    suffix = f"{mode} — {'Korea' if (iso=='KOR' or mode=='ONLINE') else iso}" + (f", PUE={pue:.2f}" if pue else "")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Energy (kWh)", f"{summary.get('total_energy_kwh',0):.3f}")
    c2.metric("Emissions (kg)", f"{summary.get('total_emissions_kg',0):.3f}")
    dur = summary.get("duration_s") or 0
    c3.metric("Duration (h)", f"{dur/3600.0:.2f}")
    tc = summary.get("total_cost_usd")
    c4.metric("Total cost (USD)", f"{tc:.2f}" if tc is not None else "–")
    st.markdown("<hr/>", unsafe_allow_html=True)

    fig = fig_white(); ax = fig.gca()
    ax.bar(epochs["epoch"], epochs["emissions_kg"].fillna(0.0))
    ax.set_xlabel("Epoch"); ax.set_ylabel("kg CO₂e")
    ax.set_title(f"Emissions per epoch — {suffix}", loc="left")
    st.pyplot(fig)

    fig = fig_white(); ax = fig.gca()
    cum = epochs["emissions_kg"].fillna(0.0).cumsum()
    ax.plot(epochs["epoch"], cum, marker="o")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cumulative kg CO₂e")
    ax.set_title(f"Cumulative emissions — {suffix}", loc="left")
    st.pyplot(fig)

    fig = fig_white(); ax = fig.gca()
    t0 = samples["timestamp"].iloc[0]; tmin = (samples["timestamp"] - t0)/60.0
    if samples["gpu_util_pct"].notna().any():
        ax.plot(tmin, samples["gpu_util_pct"].fillna(0.0), label="GPU Util (%)")
        ax.set_ylabel("GPU Utilization (%)")
    else:
        ax.plot(tmin, samples["cpu_util_pct"].fillna(0.0), label="CPU Util (%)")
        ax.set_ylabel("CPU Utilization (%)")
    ax.set_xlabel("Minutes"); ax.legend()
    ax.set_title(f"Utilization over time — {suffix}", loc="left")
    st.pyplot(fig)

# -------- Online vs Offline (Korea) --------
with tab2:
    st.subheader("Verification: ONLINE (Korea via VPN) vs OFFLINE-KOR")
    # find one offline-KOR and one online among uploaded runs
    offline = [(n,s,e) for (n,s,e,_) in runs if s.get("tracker_mode")=="offline" and s.get("country_iso_code")=="KOR"]
    online  = [(n,s,e) for (n,s,e,_) in runs if s.get("tracker_mode")=="online"]
    if not offline or not online:
        st.info("Upload at least one OFFLINE-KOR run and one ONLINE run to enable this view.")
    else:
        n_off, s_off, e_off = offline[0]
        n_on,  s_on,  e_on  = online[0]
        max_ep = min(e_off["epoch"].max(), e_on["epoch"].max())
        off = e_off[e_off["epoch"]<=max_ep]; on = e_on[e_on["epoch"]<=max_ep]

        fig = fig_white(); ax = fig.gca()
        ax.plot(off["epoch"], off["emissions_kg"].fillna(0.0), marker="o", label=f"OFFLINE KOR (PUE={s_off.get('pue')})")
        ax.plot(on["epoch"],  on["emissions_kg"].fillna(0.0),  marker="o", label=f"ONLINE auto (PUE={s_on.get('pue')})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("kg CO₂e per epoch")
        ax.set_title("Per-epoch emissions — ONLINE vs OFFLINE-KOR", loc="left")
        ax.legend(); st.pyplot(fig)

        fig = fig_white(); ax = fig.gca()
        ax.plot(off["epoch"], off["emissions_kg"].fillna(0.0).cumsum(), marker="o", label="OFFLINE KOR (cumulative)")
        ax.plot(on["epoch"],  on["emissions_kg"].fillna(0.0).cumsum(),  marker="o", label="ONLINE auto (cumulative)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cumulative kg CO₂e")
        ax.set_title("Cumulative emissions — ONLINE vs OFFLINE-KOR", loc="left")
        ax.legend(); st.pyplot(fig)

# -------- Region & PUE (what-if) --------
with tab3:
    st.subheader("Region & PUE sensitivity (fixed measured IT kWh)")
    name, summary, epochs, samples = runs[0]
    measured_kwh = summary.get("total_energy_kwh")
    if measured_kwh is None:
        st.warning("Selected run missing total_energy_kwh in summary.json")
    else:
        baseline_pue = float(summary.get("pue", 1.2))
        regions = st.multiselect("Regions", ["KOR","CAN","MEX","MNG"], default=["KOR","CAN","MEX","MNG"])
        pue_vals = st.multiselect("PUE values", [round(baseline_pue,2),1.2,1.6], default=[round(baseline_pue,2),1.2,1.6])
        EMISSION_FACTORS = {"KOR":0.45,"CAN":0.12,"MEX":0.43,"MNG":0.75}
        REGION_NAME = {"KOR":"South Korea","CAN":"Canada","MEX":"Mexico","MNG":"Mongolia"}

        rows = []
        for r in regions:
            ef = float(EMISSION_FACTORS[r])
            for p in pue_vals:
                rows.append({"iso":r, "region":REGION_NAME[r], "pue":float(p), "emissions_kg": measured_kwh*float(p)*ef})
        df = pd.DataFrame(rows).sort_values(["iso","pue"]).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

        fig = fig_white(); ax = fig.gca()
        labels = [f"{row['region']} | PUE={row['pue']}" for _, row in df.iterrows()]
        colors = [COUNTRY_COLOR[row['iso']] for _, row in df.iterrows()]
        ax.bar(labels, df["emissions_kg"], color=colors)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Emissions (kg CO₂e)")
        ax.set_title("Region & PUE sensitivity", loc="left")
        fig.tight_layout(); st.pyplot(fig)

# -------- Multi-run overlay (general) --------
with tab4:
    st.subheader("Multi-run overlay")
    if len(runs) < 2:
        st.info("Upload at least two runs to compare.")
    else:
        fig = fig_white((8.5,4.5)); ax = fig.gca()
        for name, summary, epochs, _ in runs:
            label = f"{name} | {summary.get('tracker_mode','?').upper()} | {summary.get('country_iso_code') or 'auto'} | PUE={summary.get('pue')}"
            ax.plot(epochs["epoch"], epochs["emissions_kg"].fillna(0.0), marker="o", label=label)
        ax.set_xlabel("Epoch"); ax.set_ylabel("kg CO₂e per epoch")
        ax.set_title("Per-epoch emissions — multiple runs", loc="left")
        ax.legend(fontsize=8); fig.tight_layout(); st.pyplot(fig)

        fig = fig_white((8.5,4.5)); ax = fig.gca()
        for name, summary, epochs, _ in runs:
            label = f"{name} | {summary.get('tracker_mode','?').upper()} | {summary.get('country_iso_code') or 'auto'} | PUE={summary.get('pue')}"
            ax.plot(epochs["epoch"], epochs["emissions_kg"].fillna(0.0).cumsum(), marker="o", label=label)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cumulative kg CO₂e")
        ax.set_title("Cumulative emissions — multiple runs", loc="left")
        ax.legend(fontsize=8); fig.tight_layout(); st.pyplot(fig)
