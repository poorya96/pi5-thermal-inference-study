"""
analyze.py — Thermal degradation study analysis for pi5-thermal-study

Reads all paired inference/system CSVs from results/raw_csv/, merges them
with the manual USB power log, and produces:
  - results/tables/summary_table.csv   (one row per of the 18 configs)
  - results/tables/summary_table.md    (same, markdown for paper drafting)
  - results/figures/*.png              (comparison plots)

Usage:
    python src/analyze.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw_csv"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
POWER_LOG = ROOT / "results" / "power_log.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["yolov8n", "yolo11n", "yolo12n"]
FORMAT_ORDER = ["pytorch", "onnx", "openvino"]
COOLING_ORDER = ["passive", "active"]

FILENAME_RE = re.compile(
    r"^(?P<model>yolov8n|yolo11n|yolo12n)_"
    r"(?P<format>pytorch|onnx|openvino)_"
    r"(?P<cooling>passive|active)_"
    r"(?P<run_stem>\d{8}_\d{6})_"
    r"(?P<kind>inference|system)\.csv$"
)


def discover_runs():
    """Pair up inference/system CSVs by their shared run_stem."""
    files = sorted(RAW_DIR.glob("*.csv"))
    runs = {}
    skipped = []
    for f in files:
        m = FILENAME_RE.match(f.name)
        if not m:
            skipped.append(f.name)
            continue
        key = (m["model"], m["format"], m["cooling"], m["run_stem"])
        runs.setdefault(key, {})[m["kind"]] = f

    if skipped:
        print(f"[warn] {len(skipped)} file(s) did not match the expected naming "
              f"pattern and were skipped: {skipped}")

    complete = {k: v for k, v in runs.items() if "inference" in v and "system" in v}
    incomplete = {k: v for k, v in runs.items() if k not in complete}
    if incomplete:
        print(f"[warn] {len(incomplete)} run(s) are missing one of the two CSVs "
              f"and will be skipped: {list(incomplete.keys())}")

    print(f"[info] Found {len(complete)} complete run(s) out of 18 expected.")
    return complete


def load_power_log():
    if not POWER_LOG.exists():
        print(f"[warn] {POWER_LOG} not found — power columns will be empty. "
              f"Copy your power_log.csv into results/ to include them.")
        return None
    df = pd.read_csv(POWER_LOG)
    df["watts_avg"] = df[["watts_t1", "watts_t45"]].mean(axis=1)
    return df


def build_summary(runs, power_df):
    rows = []
    for (model, fmt, cooling, run_stem), paths in runs.items():
        inf = pd.read_csv(paths["inference"])
        sysdf = pd.read_csv(paths["system"])

        row = {
            "model": model,
            "format": fmt,
            "cooling": cooling,
            "run_stem": run_stem,
            "n_frames": len(inf),
            "duration_min": round(sysdf["elapsed_seconds"].max() / 60, 1),
            "inference_ms_mean": round(inf["inference_ms"].mean(), 1),
            "inference_ms_min": round(inf["inference_ms"].min(), 1),
            "inference_ms_max": round(inf["inference_ms"].max(), 1),
            "fps_mean": round(sysdf["fps"].mean(), 2),
            "temp_start_C": round(sysdf["temperature_C"].iloc[0], 1),
            "temp_end_C": round(sysdf["temperature_C"].iloc[-1], 1),
            "temp_max_C": round(sysdf["temperature_C"].max(), 1),
            "temp_rise_C": round(sysdf["temperature_C"].iloc[-1] - sysdf["temperature_C"].iloc[0], 1),
            "cpu_freq_min_MHz": round(sysdf["cpu_freq_MHz"].min(), 0),
            "throttled": "yes" if (sysdf["throttle_status"] != "throttled=0x0").any() else "no",
            "ram_mean_MB": round(sysdf["ram_usage_MB"].mean(), 1),
            "cpu_percent_mean": round(sysdf["cpu_percent"].mean(), 1),
        }

        if power_df is not None:
            match = power_df[
                (power_df["model"] == model)
                & (power_df["format"] == fmt)
                & (power_df["cooling"] == cooling)
            ]
            if not match.empty:
                row["watts_t1"] = match["watts_t1"].iloc[0]
                row["watts_t45"] = match["watts_t45"].iloc[0]
                row["watts_avg"] = round(match["watts_avg"].iloc[0], 2)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort into a stable, paper-friendly order
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["format"] = pd.Categorical(df["format"], categories=FORMAT_ORDER, ordered=True)
    df["cooling"] = pd.Categorical(df["cooling"], categories=COOLING_ORDER, ordered=True)
    df = df.sort_values(["cooling", "model", "format"]).reset_index(drop=True)

    return df


def save_tables(summary):
    csv_path = TABLE_DIR / "summary_table.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[info] Saved {csv_path}")

    md_path = TABLE_DIR / "summary_table.md"
    try:
        with open(md_path, "w") as f:
            f.write(summary.to_markdown(index=False))
        print(f"[info] Saved {md_path}")
    except ImportError:
        print(f"[warn] 'tabulate' not installed — skipped markdown table. "
              f"Run: pip install tabulate --break-system-packages")


def load_full_system_df(runs):
    """Concatenate all system CSVs into one long dataframe for plotting."""
    frames = []
    for (model, fmt, cooling, run_stem), paths in runs.items():
        d = pd.read_csv(paths["system"])
        d["model"] = model
        d["format"] = fmt
        d["cooling"] = cooling
        d["elapsed_min"] = d["elapsed_seconds"] / 60
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_full_inference_df(runs):
    frames = []
    for (model, fmt, cooling, run_stem), paths in runs.items():
        d = pd.read_csv(paths["inference"])
        d["model"] = model
        d["format"] = fmt
        d["cooling"] = cooling
        d["elapsed_min"] = d["elapsed_seconds"] / 60
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_temp_over_time(sysdf):
    """One line per (model, format), faceted by cooling condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, cooling in zip(axes, COOLING_ORDER):
        sub = sysdf[sysdf["cooling"] == cooling]
        for model in MODEL_ORDER:
            for fmt in FORMAT_ORDER:
                line = sub[(sub["model"] == model) & (sub["format"] == fmt)]
                if line.empty:
                    continue
                ax.plot(line["elapsed_min"], line["temperature_C"],
                        label=f"{model}-{fmt}", linewidth=1.3)
        ax.set_title(f"{cooling.capitalize()} cooling")
        ax.set_xlabel("Elapsed time (min)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=9, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("CPU Temperature Over 60-Minute Runs", y=1.12)
    fig.tight_layout()
    out = FIG_DIR / "temperature_over_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[info] Saved {out}")


def plot_fps_by_format(summary):
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = range(len(MODEL_ORDER))
    for i, cooling in enumerate(COOLING_ORDER):
        for j, fmt in enumerate(FORMAT_ORDER):
            sub = summary[(summary["cooling"] == cooling) & (summary["format"] == fmt)]
            sub = sub.set_index("model").reindex(MODEL_ORDER)
            offset = (i * len(FORMAT_ORDER) + j) * width / len(FORMAT_ORDER)
    # Simpler grouped bar: one subplot per cooling condition
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bar_width = 0.25
    x = range(len(MODEL_ORDER))
    for ax, cooling in zip(axes, COOLING_ORDER):
        for i, fmt in enumerate(FORMAT_ORDER):
            sub = summary[(summary["cooling"] == cooling) & (summary["format"] == fmt)]
            sub = sub.set_index("model").reindex(MODEL_ORDER)
            positions = [p + i * bar_width for p in x]
            ax.bar(positions, sub["fps_mean"], width=bar_width, label=fmt)
        ax.set_xticks([p + bar_width for p in x])
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_title(f"{cooling.capitalize()} cooling")
        ax.set_ylabel("Mean FPS")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.suptitle("Mean FPS by Model, Format, and Cooling Condition")
    fig.tight_layout()
    out = FIG_DIR / "fps_by_format.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[info] Saved {out}")


def plot_inference_time_box(infdf):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, cooling in zip(axes, COOLING_ORDER):
        sub = infdf[infdf["cooling"] == cooling]
        groups, labels = [], []
        for model in MODEL_ORDER:
            for fmt in FORMAT_ORDER:
                vals = sub[(sub["model"] == model) & (sub["format"] == fmt)]["inference_ms"]
                if vals.empty:
                    continue
                groups.append(vals)
                labels.append(f"{model}\n{fmt}")
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.set_title(f"{cooling.capitalize()} cooling")
        ax.set_ylabel("Inference time (ms)")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Inference Time Distribution by Configuration")
    fig.tight_layout()
    out = FIG_DIR / "inference_time_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[info] Saved {out}")


def plot_passive_vs_active_temp_rise(summary):
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_width = 0.35
    x = range(len(MODEL_ORDER) * len(FORMAT_ORDER))
    labels = [f"{m}\n{f}" for m in MODEL_ORDER for f in FORMAT_ORDER]
    for i, cooling in enumerate(COOLING_ORDER):
        sub = summary[summary["cooling"] == cooling].set_index(["model", "format"])
        sub = sub.reindex([(m, f) for m in MODEL_ORDER for f in FORMAT_ORDER])
        positions = [p + i * bar_width for p in x]
        ax.bar(positions, sub["temp_rise_C"], width=bar_width, label=cooling.capitalize())
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Temperature rise over 60 min (°C)")
    ax.set_title("Temperature Rise: Passive vs Active Cooling")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG_DIR / "temp_rise_passive_vs_active.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[info] Saved {out}")


def plot_power_by_format(summary):
    if "watts_avg" not in summary.columns or summary["watts_avg"].isna().all():
        print("[warn] No power data available — skipping power plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bar_width = 0.25
    x = range(len(MODEL_ORDER))
    for ax, cooling in zip(axes, COOLING_ORDER):
        for i, fmt in enumerate(FORMAT_ORDER):
            sub = summary[(summary["cooling"] == cooling) & (summary["format"] == fmt)]
            sub = sub.set_index("model").reindex(MODEL_ORDER)
            positions = [p + i * bar_width for p in x]
            ax.bar(positions, sub["watts_avg"], width=bar_width, label=fmt)
        ax.set_xticks([p + bar_width for p in x])
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_title(f"{cooling.capitalize()} cooling")
        ax.set_ylabel("Average power (W)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.suptitle("Average Power Draw by Model, Format, and Cooling")
    fig.tight_layout()
    out = FIG_DIR / "power_by_format.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[info] Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    runs = discover_runs()
    if not runs:
        print("[error] No complete runs found in results/raw_csv/. Nothing to analyze.")
        sys.exit(1)

    power_df = load_power_log()
    summary = build_summary(runs, power_df)
    save_tables(summary)

    sysdf = load_full_system_df(runs)
    infdf = load_full_inference_df(runs)

    plot_temp_over_time(sysdf)
    plot_fps_by_format(summary)
    plot_inference_time_box(infdf)
    plot_passive_vs_active_temp_rise(summary)
    plot_power_by_format(summary)

    print("\n[info] Analysis complete.")
    print(f"[info] Tables: {TABLE_DIR}")
    print(f"[info] Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()