#!/usr/bin/env python3
"""
plot.py -- render results.csv as a bar chart.

Two panels share the version axis:
  * left  : best runtime in ms (log scale -- runtimes span orders of magnitude)
  * right : speedup vs the sequential baseline

Usage:
    python3 plot.py                       # reads results.csv -> results.png
    python3 plot.py -i other.csv -o x.png
    python3 plot.py --show                # also open an interactive window
"""
import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# Keep the canonical version order from run.py; unknown versions go last.
ORDER = ["seq", "partition", "stripe", "atomic_counter",
         "atomic_dynamic", "openmp", "omp_target", "opencl"]


def base_name(version):
    """Strip a '-u32'/'-u64' width suffix (added by run.py -w both)."""
    for suf in ("-u32", "-u64"):
        if version.endswith(suf):
            return version[:-len(suf)]
    return version


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    rows.sort(key=lambda r: ORDER.index(r["version"])
              if r["version"] in ORDER else len(ORDER))
    return rows


def plot_csv(input_path, output_path, show=False):
    """Render `input_path` (a results.csv) to `output_path`. Returns the path.

    Importable entry point so run.py can chart right after benchmarking.
    """
    if not show:
        matplotlib.use("Agg")

    rows = load(input_path)
    versions = [r["version"] for r in rows]
    best_ms = [float(r["best_ms"]) for r in rows]
    widths = [r.get("width") for r in rows]  # None for pre-width CSVs

    # Speedup baseline: the sequential run, matched BY WIDTH so a `-w both` CSV
    # compares each bar to seq of its own width (u32 vs seq-u32, u64 vs seq-u64).
    # With no seq row at all, fall back to the slowest version present. Computed
    # from best_ms so it does not depend on the speedup column.
    seq_ms_by_w = {w: float(r["best_ms"])
                   for r, w in zip(rows, widths)
                   if base_name(r["version"]) == "seq" and r["best_ms"]}
    if seq_ms_by_w:
        any_seq = next(iter(seq_ms_by_w.values()))
        speedup = [seq_ms_by_w.get(w, any_seq) / m if m else 0.0
                   for m, w in zip(best_ms, widths)]
        base_label = "seq" if len(seq_ms_by_w) == 1 else "seq (same width)"
    else:
        slow_i = max(range(len(best_ms)), key=lambda i: best_ms[i])
        base_ms, base_label = best_ms[slow_i], versions[slow_i]
        speedup = [base_ms / m if m else 0.0 for m in best_ms]

    N = int(rows[0]["N"])
    threads = rows[0].get("threads", "?")

    # Highlight the fastest (lowest runtime) version.
    fastest = min(range(len(best_ms)), key=lambda i: best_ms[i])
    base_color = "#4c72b0"
    win_color = "#dd8452"
    colors = [win_color if i == fastest else base_color for i in range(len(versions))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"copri — primes ≤ {N:,}  (threads = {threads})",
                 fontsize=14, fontweight="bold")

    # --- left: runtime (log) -------------------------------------------------
    b1 = ax1.bar(versions, best_ms, color=colors)
    ax1.set_yscale("log")
    ax1.set_ylabel("best runtime [ms]  (log scale)")
    ax1.set_title("Runtime — lower is better")
    ax1.bar_label(b1, labels=[f"{v:,.1f}" for v in best_ms],
                  padding=3, fontsize=8)
    ax1.grid(axis="y", which="both", ls=":", alpha=0.4)

    # --- right: speedup ------------------------------------------------------
    b2 = ax2.bar(versions, speedup, color=colors)
    ax2.set_ylabel(f"speedup vs {base_label}  (×)")
    ax2.set_title(f"Speedup vs {base_label} — higher is better")
    ax2.bar_label(b2, labels=[f"{v:.2f}×" for v in speedup],
                  padding=3, fontsize=8)
    ax2.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    ax2.grid(axis="y", ls=":", alpha=0.4)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=str(ROOT / "results.csv"))
    ap.add_argument("-o", "--output", default=str(ROOT / "results.png"))
    ap.add_argument("--show", action="store_true", help="open a window too")
    args = ap.parse_args()
    plot_csv(args.input, args.output, show=args.show)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
