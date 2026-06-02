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
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "bin"
IS_WINDOWS = os.name == "nt"

if IS_WINDOWS:
    # Windows consoles/pipes default to cp1252, which can't encode the non-ASCII
    # characters our tables and plot titles use (e.g. "≤"). Force UTF-8 so a
    # stray print never aborts a long run. Imported by sweep.py/scale.py too.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def bin_path(name):
    """Path to a built binary, adding the .exe suffix on Windows."""
    return BIN / (name + ".exe") if IS_WINDOWS else BIN / name

# Canonical order in which to present versions (others appended alphabetically).
ORDER = ["seq", "partition", "stripe", "atomic_counter",
         "atomic_dynamic", "openmp", "omp_target", "opencl",
         "sieve_cpu", "sieve_gpu", "sieve_gpu_barrett"]

# Mirror of kU32SafeMax in common/prime.hpp: above this, the uint32 path is
# unsafe, so --width both / 32 skip u32 and fall back to u64.
U32_SAFE_MAX = 4_000_000_000


def discover_binaries():
    if not BIN.is_dir():
        return []
    if IS_WINDOWS:
        # Windows has no executable bit; match bin\*.exe and drop the suffix.
        found = {p.stem for p in BIN.iterdir()
                 if p.is_file() and p.suffix.lower() == ".exe"}
    else:
        found = {p.name for p in BIN.iterdir()
                 if p.is_file() and p.stat().st_mode & 0o111}
    ordered = [v for v in ORDER if v in found]
    ordered += sorted(found - set(ORDER))
    return ordered


def _msvc_env(vcvars):
    """Run vcvars64.bat and capture the environment it sets (so cl.exe, the
    Windows SDK, and CUDA_PATH are visible to cmake). Returns an env dict or None.
    """
    # shell=True so cmd parses the quoted path itself; passing this as a list
    # element would let Python backslash-escape the quotes, which cmd mishandles.
    out = subprocess.run(f'"{vcvars}" >nul 2>&1 && set',
                         capture_output=True, text=True, shell=True)
    if out.returncode != 0:
        return None
    env = {}
    for line in out.stdout.splitlines():
        key, sep, val = line.partition("=")
        if sep:
            env[key] = val
    return env or None


def build_windows():
    """Configure + build via CMake inside the MSVC environment.

    Works from any shell: locate vcvars64.bat plus the cmake/ninja bundled with
    Visual Studio, capture the MSVC environment, and drive CMakeLists.txt with it.
    Returns True on success; otherwise tell the user to build manually and re-run
    with --no-build.
    """
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    vs = ""
    if vswhere.exists():
        vs = subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True).stdout.strip()
    if not vs:
        print("Visual Studio C++ toolset not found. Build manually with cmake, "
              "then re-run with --no-build.", file=sys.stderr)
        return False

    vsp = Path(vs)
    vcvars = vsp / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    cmake = (vsp / "Common7" / "IDE" / "CommonExtensions" / "Microsoft"
             / "CMake" / "CMake" / "bin" / "cmake.exe")
    ninja = (vsp / "Common7" / "IDE" / "CommonExtensions" / "Microsoft"
             / "CMake" / "Ninja" / "ninja.exe")
    if not cmake.exists():
        cmake = Path(shutil.which("cmake") or "cmake")

    env = _msvc_env(vcvars)
    if env is None:
        print(f"could not initialize MSVC environment ({vcvars})", file=sys.stderr)
        return False

    # Make nvcc discoverable so CMake's default GPU backend (CUDA when present)
    # works from any shell, even one launched before the CUDA Toolkit install.
    cuda_path = env.get("CUDA_PATH")
    if not cuda_path:
        roots = sorted(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA").glob("v*"))
        cuda_path = str(roots[-1]) if roots else None
    if cuda_path:
        env["PATH"] = str(Path(cuda_path) / "bin") + os.pathsep + env.get("PATH", "")

    build_dir = ROOT / "build"
    configure = [str(cmake), "-S", str(ROOT), "-B", str(build_dir),
                 "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"]
    if ninja.exists():
        configure.append(f"-DCMAKE_MAKE_PROGRAM={ninja}")
    build = [str(cmake), "--build", str(build_dir)]

    print("building (cmake + ninja via MSVC)...")
    for step in (configure, build):
        if subprocess.run(step, env=env).returncode != 0:
            print("build failed", file=sys.stderr)
            return False
    return True


def build_default():
    """Build every version for the current platform (Windows: CMake/MSVC; else
    `make`). Shared by run.py, sweep.py and scale.py. Returns True on success."""
    if IS_WINDOWS:
        return build_windows()
    if shutil.which("make") is None:
        print("make not found; use --no-build", file=sys.stderr)
        return False
    print("building (make)...")
    return subprocess.run(["make"], cwd=ROOT).returncode == 0


def run_one(name, n, threads, repeats, width="auto"):
    """Run bin/<name> `repeats` times.

    Returns (count, [times_ms], width_bits) or None. `width` is one of
    "auto"/"u32"/"u64" and is passed through to the binary.
    """
    cmd = [str(bin_path(name)), str(n)]
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

    if not args.no_build and not build_default():
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
