#!/usr/bin/env python3
"""
sweep.py -- thread-scaling sweep.

Where run.py fixes the thread count and varies the VERSION, this fixes N and
varies the THREAD COUNT for each parallel version, to answer "how well does
each approach scale as cores are added?".

For every version x thread-count it records the best runtime, then derives:
  * speedup_self -- best_ms(1 thread) / best_ms(t threads)   (vs its own 1-thread run)
  * efficiency   -- speedup_self / t                          (1.0 == perfect scaling)

Outputs a tidy CSV (one row per version x thread count) and a two-panel line
plot: speedup vs threads (with an ideal y=x line) and parallel efficiency.

Examples
--------
    python3 sweep.py                          # N=10^7, threads 1,2,4,8,12,16
    python3 sweep.py -n 50000000 -p 1,2,4,8,16
    python3 sweep.py --versions stripe atomic_dynamic
    python3 sweep.py --no-build --no-plot

seq (always 1 thread) and opencl (uses GPU work-items, not threads) are excluded
by default -- a thread sweep does not apply to them.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Imported so the actual run/parse/build logic lives in exactly one place.
from run import BIN, run_one, discover_binaries, build_default

# Versions whose thread count is meaningfully controllable via argv[2].
SWEEPABLE = ["partition", "stripe", "atomic_counter", "atomic_dynamic",
             "openmp", "sieve_cpu"]


def parse_threads(spec):
    vals = sorted({int(x) for x in spec.split(",") if x.strip()})
    if not vals:
        raise SystemExit("no thread counts parsed from --points")
    return vals


def make_plot(rows, threads, versions, n, output, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"copri — thread scaling, primes ≤ {n:,}",
                 fontsize=14, fontweight="bold")

    by_ver = {v: {r["threads"]: r for r in rows if r["version"] == v}
              for v in versions}

    # Ideal linear reference on the speedup panel.
    ax1.plot(threads, threads, ls="--", color="grey", lw=1.2,
             label="ideal (linear)")

    for v in versions:
        pts = by_ver[v]
        xs = [t for t in threads if t in pts]
        sp = [pts[t]["speedup_self"] for t in xs]
        ef = [pts[t]["efficiency"] for t in xs]
        ax1.plot(xs, sp, marker="o", label=v)
        ax2.plot(xs, ef, marker="o", label=v)

    ax1.set_title("Speedup vs own 1-thread run — higher is better")
    ax1.set_xlabel("threads")
    ax1.set_ylabel("speedup (×)")
    ax1.set_xticks(threads)
    ax1.grid(ls=":", alpha=0.4)
    ax1.legend(fontsize=8)

    ax2.set_title("Parallel efficiency — 1.0 is perfect")
    ax2.set_xlabel("threads")
    ax2.set_ylabel("efficiency = speedup / threads")
    ax2.set_xticks(threads)
    ax2.axhline(1.0, ls="--", color="grey", lw=1.2, alpha=0.7)
    ax2.set_ylim(0, 1.15)
    ax2.grid(ls=":", alpha=0.4)
    ax2.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--limit", type=int, default=10_000_000,
                    help="upper limit N (inclusive). Default 10^7.")
    ap.add_argument("-p", "--points", default="1,2,4,8,12,16",
                    help="comma-separated thread counts. Default 1,2,4,8,12,16.")
    ap.add_argument("-r", "--repeats", type=int, default=3,
                    help="runs per (version, threads); best is kept.")
    ap.add_argument("--versions", nargs="+", metavar="VER",
                    help="versions to sweep (default: all sweepable in bin/).")
    ap.add_argument("-o", "--output", default=None,
                    help="output CSV path. Default: sweep-<timestamp>.csv, "
                         "so a run never overwrites an earlier one.")
    ap.add_argument("--plot", default=None, metavar="PNG",
                    help="plot path (default: alongside the CSV as .png).")
    ap.add_argument("--no-plot", action="store_true", help="skip the chart.")
    ap.add_argument("--no-build", action="store_true",
                    help="do not run `make` first.")
    ap.add_argument("--show", action="store_true", help="open a window too.")
    args = ap.parse_args()

    # Timestamp the default output so long runs are never clobbered. An explicit
    # -o is honored as given.
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output = str(ROOT / f"sweep-{stamp}.csv")

    threads = parse_threads(args.points)
    if 1 not in threads:
        print("note: 1 not in --points, so speedup is relative to the smallest "
              f"thread count ({threads[0]}).", file=sys.stderr)

    if not args.no_build and not build_default():
        return 1

    built = set(discover_binaries())
    want = args.versions if args.versions else SWEEPABLE
    versions = [v for v in want if v in built and v in SWEEPABLE]
    skipped = [v for v in want if v not in versions]
    if skipped:
        print(f"skipping (not built or not sweepable): {', '.join(skipped)}")
    if not versions:
        print("no sweepable versions available", file=sys.stderr)
        return 1

    print(f"\nN = {args.limit:,}   repeats = {args.repeats}   "
          f"threads = {threads}")
    print(f"versions: {', '.join(versions)}\n")

    rows = []
    for v in versions:
        baseline = None
        for t in threads:
            print(f"running {v} @ {t} thread(s) ...", flush=True)
            out = run_one(v, args.limit, t, args.repeats)
            if out is None:
                continue
            count, times, _ = out
            best = round(min(times), 3)
            if baseline is None:
                baseline = best  # first (smallest) thread count = self-baseline
            rows.append({
                "version": v,
                "N": args.limit,
                "threads": t,
                "repeats": len(times),
                "count": count,
                "best_ms": best,
                "speedup_self": round(baseline / best, 3) if best else "",
                "efficiency": round(baseline / best / t, 3) if best else "",
            })

    if not rows:
        print("no successful runs", file=sys.stderr)
        return 1

    # Correctness: counts must agree within each version (and ideally overall).
    counts = {r["count"] for r in rows}
    if len(counts) == 1:
        print(f"\nall runs agree: {next(iter(counts)):,} primes ≤ {args.limit:,}")
    else:
        print("\n*** WARNING: prime counts disagree across runs ***")

    # Console table.
    print()
    hdr = f"{'version':<16}{'threads':>8}{'best_ms':>12}{'speedup':>10}{'eff':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['version']:<16}{r['threads']:>8}{r['best_ms']:>12.3f}"
              f"{r['speedup_self']:>9}x{r['efficiency']:>8}")

    fields = ["version", "N", "threads", "repeats", "count",
              "best_ms", "speedup_self", "efficiency"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.output}")

    if not args.no_plot:
        png = args.plot or str(Path(args.output).with_suffix(".png"))
        try:
            make_plot(rows, threads, versions, args.limit, png, args.show)
            print(f"wrote {png}")
        except Exception as e:  # plotting is a convenience, never fail the run
            print(f"plot skipped: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
