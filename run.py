#!/usr/bin/env python3
"""
run.py -- run every built prime-counting version at the same upper limit and
write the measured runtimes to a CSV.

Examples
--------
    python3 run.py                      # N=1_000_000, all versions in bin/
    python3 run.py -n 10000000          # N=10^7
    python3 run.py -n 5000000 -r 5      # 5 repeats, keep best & median
    python3 run.py -t 8                 # force 8 threads for CPU versions
    python3 run.py --only seq stripe    # subset
    python3 run.py --no-build           # skip `make`, run whatever is in bin/

The runner builds first (via `make`), runs each binary, parses its one-line JSON
result, verifies every version agrees on the prime count, prints a table, and
writes results.csv (override with -o).
"""
import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "bin"

# Canonical order in which to present versions (others appended alphabetically).
ORDER = ["seq", "partition", "stripe", "atomic_counter",
         "atomic_dynamic", "openmp", "opencl"]


def discover_binaries():
    if not BIN.is_dir():
        return []
    found = {p.name for p in BIN.iterdir()
             if p.is_file() and p.stat().st_mode & 0o111}
    ordered = [v for v in ORDER if v in found]
    ordered += sorted(found - set(ORDER))
    return ordered


def run_one(name, n, threads, repeats):
    """Run bin/<name> `repeats` times; return (count, [times_ms]) or None."""
    cmd = [str(BIN / name), str(n)]
    if threads:
        cmd.append(str(threads))
    times, count = [], None
    for _ in range(repeats):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ! {name} failed (exit {e.returncode}):\n{e.stderr.strip()}",
                  file=sys.stderr)
            return None
        line = next((l for l in proc.stdout.splitlines() if l.startswith("{")), None)
        if line is None:
            print(f"  ! {name}: no JSON result on stdout", file=sys.stderr)
            return None
        rec = json.loads(line)
        times.append(rec["time_ms"])
        count = rec["count"]
    return count, times


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--limit", type=int, default=1_000_000,
                    help="upper limit N (inclusive). Default 1_000_000.")
    ap.add_argument("-t", "--threads", type=int, default=0,
                    help="thread count for CPU versions (0 = auto).")
    ap.add_argument("-r", "--repeats", type=int, default=3,
                    help="runs per version; best & median are recorded.")
    ap.add_argument("-o", "--output", default=None,
                    help="output CSV path. Default: results-<timestamp>.csv, "
                         "so a run never overwrites an earlier one.")
    ap.add_argument("--only", nargs="+", metavar="VER",
                    help="only run these versions.")
    ap.add_argument("--skip", nargs="+", metavar="VER", default=[],
                    help="skip these versions.")
    ap.add_argument("--no-build", action="store_true",
                    help="do not run `make` before benchmarking.")
    ap.add_argument("--plot", nargs="?", const=True, default=False,
                    metavar="PNG",
                    help="after benchmarking, render a bar chart. Optionally "
                         "give a path (default: results.png next to the CSV).")
    args = ap.parse_args()

    # Timestamp the default output so long runs are never clobbered. An explicit
    # -o is honored as given.
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output = str(ROOT / f"results-{stamp}.csv")

    if not args.no_build:
        if shutil.which("make") is None:
            print("make not found; use --no-build", file=sys.stderr)
            return 1
        print("building (make)...")
        if subprocess.run(["make"], cwd=ROOT).returncode != 0:
            print("build failed", file=sys.stderr)
            return 1

    versions = discover_binaries()
    if args.only:
        versions = [v for v in versions if v in args.only]
    versions = [v for v in versions if v not in args.skip]
    if not versions:
        print("no binaries to run (did the build produce bin/?)", file=sys.stderr)
        return 1

    print(f"\nN = {args.limit:,}   repeats = {args.repeats}   "
          f"threads = {args.threads or 'auto'}")
    print(f"versions: {', '.join(versions)}\n")

    results = []
    for v in versions:
        print(f"running {v} ...", flush=True)
        out = run_one(v, args.limit, args.threads, args.repeats)
        if out is None:
            continue
        count, times = out
        results.append({
            "version": v,
            "N": args.limit,
            "threads": args.threads or "auto",
            "repeats": len(times),
            "count": count,
            "best_ms": round(min(times), 3),
            "median_ms": round(statistics.median(times), 3),
        })

    if not results:
        print("no successful runs", file=sys.stderr)
        return 1

    # Correctness check: every version must agree on the prime count.
    counts = {r["version"]: r["count"] for r in results}
    if len(set(counts.values())) > 1:
        print("\n*** WARNING: prime counts DISAGREE across versions ***")
        for ver, c in counts.items():
            print(f"    {ver}: {c}")
    else:
        print(f"\nall versions agree: {next(iter(counts.values())):,} primes <= {args.limit:,}")

    # Speedup relative to the sequential baseline (if present).
    base = next((r["best_ms"] for r in results if r["version"] == "seq"), None)
    for r in results:
        r["speedup_vs_seq"] = round(base / r["best_ms"], 2) if base else ""

    # Pretty table.
    print()
    hdr = f"{'version':<16}{'best_ms':>12}{'median_ms':>12}{'speedup':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        sp = f"{r['speedup_vs_seq']}x" if r["speedup_vs_seq"] != "" else "-"
        print(f"{r['version']:<16}{r['best_ms']:>12.3f}"
              f"{r['median_ms']:>12.3f}{sp:>10}")

    # CSV.
    fields = ["version", "N", "threads", "repeats", "count",
              "best_ms", "median_ms", "speedup_vs_seq"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {args.output}")

    if args.plot:
        png = args.plot if isinstance(args.plot, str) \
            else str(Path(args.output).with_suffix(".png"))
        try:
            from plot import plot_csv
            plot_csv(args.output, png)
            print(f"wrote {png}")
        except Exception as e:  # plotting is a convenience, never fail the run
            print(f"plot skipped: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
