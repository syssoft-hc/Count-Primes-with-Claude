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
         "atomic_dynamic", "openmp", "omp_target", "opencl",
         "sieve_cpu", "sieve_gpu"]

# Mirror of kU32SafeMax in common/prime.hpp: above this, the uint32 path is
# unsafe, so --width both / 32 skip u32 and fall back to u64.
U32_SAFE_MAX = 4_000_000_000


def discover_binaries():
    if not BIN.is_dir():
        return []
    found = {p.name for p in BIN.iterdir()
             if p.is_file() and p.stat().st_mode & 0o111}
    ordered = [v for v in ORDER if v in found]
    ordered += sorted(found - set(ORDER))
    return ordered


def run_one(name, n, threads, repeats, width="auto"):
    """Run bin/<name> `repeats` times.

    Returns (count, [times_ms], width_bits) or None. `width` is one of
    "auto"/"u32"/"u64" and is passed through to the binary.
    """
    cmd = [str(BIN / name), str(n)]
    if threads:
        cmd.append(str(threads))
    if width and width != "auto":
        cmd.append(width)
    times, count, wbits = [], None, None
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
        wbits = rec.get("width")
    return count, times, wbits


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
    ap.add_argument("-w", "--width", choices=["auto", "32", "64", "both"],
                    default="auto",
                    help="integer width for is_prime: auto picks by N (u32 if "
                         "N<=4e9, else u64); both runs each version at u32 AND "
                         "u64 (labelled <ver>-u32 / -u64).")
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

    # Build the (label, base_version, width_arg) jobs. In "both" mode each
    # version runs twice (u32 only if N is within the safe range).
    def jobs_for(v):
        if args.width == "both":
            js = []
            if args.limit <= U32_SAFE_MAX:
                js.append((f"{v}-u32", v, "u32"))
            js.append((f"{v}-u64", v, "u64"))
            return js
        warg = {"auto": "auto", "32": "u32", "64": "u64"}[args.width]
        return [(v, v, warg)]

    if args.width in ("32", "both") and args.limit > U32_SAFE_MAX:
        print(f"note: N>{U32_SAFE_MAX:,} exceeds the uint32 safe limit; "
              f"u32 runs are skipped (would be unsafe).", file=sys.stderr)

    jobs = [j for v in versions for j in jobs_for(v)]

    print(f"\nN = {args.limit:,}   repeats = {args.repeats}   "
          f"threads = {args.threads or 'auto'}   width = {args.width}")
    print(f"runs: {', '.join(label for label, _, _ in jobs)}\n")

    results = []
    for label, base, warg in jobs:
        print(f"running {label} ...", flush=True)
        out = run_one(base, args.limit, args.threads, args.repeats, warg)
        if out is None:
            continue
        count, times, wbits = out
        results.append({
            "version": label,
            "base": base,
            "N": args.limit,
            "threads": args.threads or "auto",
            "width": wbits,
            "repeats": len(times),
            "count": count,
            "best_ms": round(min(times), 3),
            "median_ms": round(statistics.median(times), 3),
        })

    if not results:
        print("no successful runs", file=sys.stderr)
        return 1

    # Correctness check: every run must agree on the prime count.
    counts = {r["version"]: r["count"] for r in results}
    if len(set(counts.values())) > 1:
        print("\n*** WARNING: prime counts DISAGREE across runs ***")
        for ver, c in counts.items():
            print(f"    {ver}: {c}")
    else:
        print(f"\nall runs agree: {next(iter(counts.values())):,} primes <= {args.limit:,}")

    # Speedup vs the sequential baseline of the SAME width (so u32 is compared to
    # seq-u32 and u64 to seq-u64).
    seq_by_width = {r["width"]: r["best_ms"] for r in results if r["base"] == "seq"}
    for r in results:
        b = seq_by_width.get(r["width"])
        r["speedup_vs_seq"] = round(b / r["best_ms"], 2) if b else ""

    # Pretty table.
    print()
    hdr = f"{'version':<18}{'width':>6}{'best_ms':>12}{'median_ms':>12}{'speedup':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        sp = f"{r['speedup_vs_seq']}x" if r["speedup_vs_seq"] != "" else "-"
        print(f"{r['version']:<18}{('u'+str(r['width'])):>6}{r['best_ms']:>12.3f}"
              f"{r['median_ms']:>12.3f}{sp:>10}")

    # CSV.
    fields = ["version", "N", "threads", "width", "repeats", "count",
              "best_ms", "median_ms", "speedup_vs_seq"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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
