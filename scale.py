#!/usr/bin/env python3
"""
scale.py -- problem-size scaling sweep.

Where sweep.py fixes N and varies the thread count, this fixes the versions and
varies the PROBLEM SIZE N = 10^n, to show how the GPU-vs-CPU trade-off changes
with scale. Built for the two sieves (sieve_cpu vs sieve_gpu): the GPU loses at
small N (kernel-launch overhead) and should pull ahead once N is large enough to
amortize it -- this plots exactly where that crossover happens.

For every version x size it records the best runtime, verifies the prime count
against the known pi(10^n), and derives the GPU-over-CPU speedup per size.

Outputs a tidy CSV (one row per version x size) and a two-panel plot:
  * left  : runtime vs N (log-log), one line per version
  * right : sieve_gpu speedup over sieve_cpu vs N, with a y=1 crossover line

Examples
--------
    python3 scale.py                       # sieve_cpu vs sieve_gpu, n=3..11
    python3 scale.py --exp 3-12            # ... up to 10^12 (slow: ~minutes)
    python3 scale.py --versions sieve_gpu sieve_cpu -r 5
"""
import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from run import run_one, discover_binaries  # reuse the runner

# Known prime counts pi(10^n), for the correctness check.
PI = {3: 168, 4: 1229, 5: 9592, 6: 78498, 7: 664579, 8: 5761455,
      9: 50847534, 10: 455052511, 11: 4118054813, 12: 37607912018}


def parse_exp(spec):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted({int(x) for x in spec.split(",") if x.strip()})


def make_plot(rows, exps, versions, output, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("copri — sieve scaling: CPU vs GPU", fontsize=14, fontweight="bold")

    by = {v: {r["n"]: r for r in rows if r["version"] == v} for v in versions}
    Ns = [10 ** n for n in exps]

    # --- left: runtime vs N (log-log) ---------------------------------------
    for v in versions:
        xs = [10 ** n for n in exps if n in by[v]]
        ys = [by[v][n]["best_ms"] for n in exps if n in by[v]]
        ax1.plot(xs, ys, marker="o", label=v)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("N (upper limit)"); ax1.set_ylabel("best runtime [ms]")
    ax1.set_title("Runtime vs problem size — lower is better")
    ax1.grid(which="both", ls=":", alpha=0.4); ax1.legend()

    # --- right: GPU-over-CPU speedup vs N -----------------------------------
    cpu, gpu = by.get("sieve_cpu", {}), by.get("sieve_gpu", {})
    sx, sy = [], []
    for n in exps:
        if n in cpu and n in gpu and gpu[n]["best_ms"]:
            sx.append(10 ** n)
            sy.append(cpu[n]["best_ms"] / gpu[n]["best_ms"])
    if sx:
        ax2.plot(sx, sy, marker="o", color="#dd8452")
        ax2.axhline(1.0, ls="--", color="grey", lw=1.2, alpha=0.8)
        ax2.fill_between(sx, 1.0, sy, where=[y >= 1 for y in sy],
                         color="#dd8452", alpha=0.15)
        ax2.set_xscale("log")
        ax2.set_xlabel("N (upper limit)")
        ax2.set_ylabel("sieve_gpu speedup over sieve_cpu  (×)")
        ax2.set_title("GPU vs CPU — above 1.0 the GPU wins")
        ax2.grid(which="both", ls=":", alpha=0.4)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["sieve_cpu", "sieve_gpu"],
                    help="versions to scale (default: the two sieves).")
    ap.add_argument("--exp", default="3-11",
                    help="exponents n for N=10^n: range 'a-b' or list 'a,b,c'.")
    ap.add_argument("-r", "--repeats", type=int, default=3)
    ap.add_argument("-o", "--output", default=None,
                    help="CSV path. Default: scale-<timestamp>.csv.")
    ap.add_argument("--plot", default=None, metavar="PNG")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output = str(ROOT / f"scale-{stamp}.csv")
    exps = parse_exp(args.exp)

    if not args.no_build:
        if shutil.which("make") is None:
            print("make not found; use --no-build", file=sys.stderr); return 1
        print("building (make)...")
        if subprocess.run(["make"], cwd=ROOT).returncode != 0:
            print("build failed", file=sys.stderr); return 1

    built = set(discover_binaries())
    versions = [v for v in args.versions if v in built]
    missing = [v for v in args.versions if v not in built]
    if missing:
        print(f"skipping (not built): {', '.join(missing)}")
    if not versions:
        print("no versions to run", file=sys.stderr); return 1

    print(f"\nversions: {', '.join(versions)}   sizes: "
          f"{', '.join('10^%d' % n for n in exps)}   repeats = {args.repeats}\n")

    rows, ok = [], True
    for n in exps:
        N = 10 ** n
        for v in versions:
            print(f"running {v} @ N=10^{n} ...", flush=True)
            out = run_one(v, N, 0, args.repeats)
            if out is None:
                continue
            count, times, _ = out
            if n in PI and count != PI[n]:
                print(f"  *** WRONG: {v} N=10^{n} -> {count}, expected {PI[n]}",
                      file=sys.stderr)
                ok = False
            rows.append({"n": n, "N": N, "version": v,
                         "best_ms": round(min(times), 3),
                         "median_ms": round(sorted(times)[len(times) // 2], 3),
                         "count": count})

    if not rows:
        print("no successful runs", file=sys.stderr); return 1
    print("\nall prime counts verified against known pi(10^n)." if ok
          else "\n*** some counts were WRONG (see above) ***")

    # Console table: GPU speedup over CPU per size.
    cpu = {r["n"]: r["best_ms"] for r in rows if r["version"] == "sieve_cpu"}
    gpu = {r["n"]: r["best_ms"] for r in rows if r["version"] == "sieve_gpu"}
    print(f"\n{'N':>8}{'sieve_cpu':>14}{'sieve_gpu':>14}{'gpu/cpu':>10}")
    print("-" * 46)
    for n in exps:
        c, g = cpu.get(n), gpu.get(n)
        sp = f"{c / g:.2f}x" if c and g else "-"
        cs = f"{c:.3f}" if c else "-"
        gs = f"{g:.3f}" if g else "-"
        print(f"10^{n:<5}{cs:>14}{gs:>14}{sp:>10}")

    fields = ["n", "N", "version", "best_ms", "median_ms", "count"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.output}")

    if not args.no_plot:
        png = args.plot or str(Path(args.output).with_suffix(".png"))
        try:
            make_plot(rows, exps, versions, png, args.show)
            print(f"wrote {png}")
        except Exception as e:
            print(f"plot skipped: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
