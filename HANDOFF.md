# HANDOFF — picking this project up on another machine

> **If you are a fresh Claude session (especially on the Windows/CUDA machine): read
> this file first.** The previous work happened on an Apple M3 Max. Claude's
> file-based *memory* lives in `~/.claude/projects/<slug>/memory/`, **outside the
> repo**, so it did **not** travel with a `git clone` or USB copy. This file is the
> portable replacement for that memory. For the full story and numbers, also read
> [`README.md`](README.md) and the travelogue
> [`A-Random-Walk-to-a-Trillion.md`](A-Random-Walk-to-a-Trillion.md). Verbatim copies
> of the old memory notes are in [`docs/claude-memory/`](docs/claude-memory/).

---

## 0. STATUS — Windows + CUDA port COMPLETE (2026-06-02)

The port planned in §6 is **done and validated** on the Windows box (Intel
i9-7900X, 10c/20t, 64 GB; **NVIDIA RTX 2080 Ti**, sm_75; CUDA 13.3, MSVC 14.4,
Python 3.13). Summary of what changed (all additive — the macOS `Makefile`,
`common/`, and `#ifdef __APPLE__` paths are untouched):

- **Build:** new `CMakeLists.txt`. GPU versions build from CUDA `.cu` on Windows
  and OpenCL on macOS under the **same binary names**; select with
  `-DCOPRI_GPU_BACKEND=cuda|opencl|off` (default: CUDA if a CUDA compiler is
  found, else OpenCL). `openmp`/`omp_target` are not built on Windows.
- **CUDA sources:** `src/{opencl,sieve_gpu,sieve_gpu_barrett}.cu` +
  `src/prime_kernel.cuh`, `src/sieve_kernel.cuh`, `src/cuda_util.cuh` — direct
  ports of the OpenCL kernels per the §6C table (get_global_id →
  blockIdx*blockDim+threadIdx, __local → extern __shared__, barrier →
  __syncthreads, mul_hi → __umul64hi). nvcc compiles kernels at build time, so
  there is no runtime KERNEL_PATH.
- **Tooling:** `run.py`/`sweep.py`/`scale.py` are cross-platform — discover
  `bin\*.exe`, build via a shared `build_default()` that captures the MSVC
  environment and drives cmake/ninja from any shell, and force UTF-8 stdout on
  Windows. Three portable C++ fixes (MSVC C3493 lambda captures in
  stripe/atomic_counter/atomic_dynamic).
- **Results:** `results_rtx2080ti/` — `run_10e8`, `sweep_10e7`,
  `scale_sieve_3-11` (.csv + .png). All counts verified against the π(10ⁿ) oracle.

**Headline numbers (RTX 2080 Ti, CUDA):**
- N=10⁸: `opencl` (CUDA trial division) 691 ms = **69×** over `seq`;
  `sieve_gpu_barrett` 2.5 ms (~19,000× over `seq`).
- N=10⁹: `opencl` 21.4 s (vs M3 Max OpenCL 49.8 s); `sieve_gpu_barrett` 19.2 ms
  vs `sieve_cpu` 90 ms (~4.7×).
- Sieve scaling 10³–10¹¹: the GPU sieve **never reverses** here — 5–11× over the
  CPU sieve across the whole range (~7.3× at 10¹¹), unlike the M3 Max where plain
  `sieve_gpu` fell to 0.75× at 10¹¹. Discrete VRAM + a native 64-bit integer
  divide remove the Apple GPU's wall, as predicted below. Barrett still wins, just
  by a smaller margin than on Apple.

**Not done yet:**
- **`results_rtx2080ti/SPECS.md` — please create this on the Windows box.** The Mac
  side now has [`results_m3max/SPECS.md`](results_m3max/SPECS.md) documenting that
  machine's hardware + software environment; mirror it for the RTX box so the two
  sets are comparable. Use the **same structure** (Machine / CPU / GPU / Memory /
  OS / Toolchain + build flags) and capture values **live on the machine** (not from
  this file): `wmic`/Task Manager or `Get-ComputerInfo` for CPU/RAM, `nvidia-smi`
  for the GPU + driver, `nvcc --version` for the CUDA Toolkit, `cl.exe`/MSVC version,
  `cmake --version`, `python --version` + matplotlib/numpy. Note the mirror-image
  hardware fact: the **RTX 2080 Ti *has* a fast native 64-bit integer divide** (sm_75),
  which is why Barrett is only a nice-to-have here — the opposite of the Apple note.
- OpenCL-vs-CUDA on the *same* RTX (the CUDA build replaces the OpenCL binaries
  under identical names; build `-DCOPRI_GPU_BACKEND=opencl` separately to get both,
  ideally under `*_ocl` labels).

Also new since the port: [`COMPARE-RESULTS.md`](COMPARE-RESULTS.md) (cross-platform
M3 Max vs RTX comparison) and an `opencl`-label-is-CUDA caveat now documented in the
README, travelogue, and that file. Branch `windows-cuda-port` has been merged to
`master` and pushed.

---

## 1. How to transfer & resume

- **Preferred:** on the Windows box, `git clone git@github.com:syssoft-hc/Count-Primes-with-Claude.git`
  (remote name `github`, branch `master`). Cleaner than USB — skips the stale Mac
  `bin/` and `.claude/`.
- **USB is fine too**, but after copying: delete/ignore the Mac `bin/` (Mach-O
  binaries won't run on Windows) and rebuild from source. `.git/` carries the full
  history, so commits/remote survive the copy.
- The plan: do the Windows/CUDA port in a **fresh session on the Windows machine**,
  set up the toolchain there, build, run, save results into a new `results_<machine>/`
  dir, then **push to GitHub** so the Mac and Windows results live together.

## 2. What this project is (one paragraph)

A didactic project: **count the primes ≤ N many parallel ways on a single host** to
study how to exploit parallelism. Same trivial trial-division `is_prime` in every
"fair" version (template `is_prime_impl<T>` → `is_prime_uint32/64` in
`common/prime.hpp`); two sieve versions deliberately break that rule. Tooling:
`run.py` (per-version runner → CSV, `-w` width, `--plot`), `plot.py`, `sweep.py`
(thread scaling), `scale.py` (problem-size scaling). Snapshots are kept per machine
in `results_<machine>/` (e.g. `results_m3max/`); the naming convention (underscore =
tracked snapshot, hyphen = git-ignored throwaway) is described in the README.

**Versions:** `seq`, `partition`, `stripe`, `atomic_counter`, `atomic_dynamic`
(std::thread); `openmp`, `omp_target` (libomp, optional); `opencl` (GPU trial
division); `sieve_cpu`, `sieve_gpu`, `sieve_gpu_barrett` (segmented sieve — a
different algorithm). All print one JSON line to stdout that `run.py` parses.

## 3. The lessons found so far (the arc, condensed)

1. Dynamic work-stealing beats static partitioning (~11× on 16 cores).
2. **Striping trap:** a cyclic stride must be coprime to the data — an even stride
   sends all evens to one thread (no speedup at 2 threads).
3. **uint32 vs uint64** swings the GPU ~12× (Apple GPU has no native 64-bit integer
   divide); ~1× on the ARM CPU.
4. **Algorithm ≫ parallelism:** the sieve is ~1000× faster than trial division.
5. **Unified memory rewards on-chip blocking:** the first GPU sieve *lost* by
   streaming a 500 MB array through the RAM the cache-blocked CPU uses; blocking in
   `__local` memory flipped it to a win.
6. **GPU sieve sweet spot → reversal → fix:** `sieve_gpu` beat `sieve_cpu` only at
   ~10⁹–10¹⁰, then lost at 10¹¹–10¹² (per-segment 64-bit `start % p`, base primes
   grow as √N). **`sieve_gpu_barrett`** replaced that division with **Barrett
   reduction** (`mul_hi` + multiply + subtract, `μ=⌊2⁶⁴/p⌋` precomputed on host) and
   reclaimed large N: ~2× over the CPU through 10¹².
7. Recurring villain throughout: **64-bit integer division** — beaten by trading
   divide for multiply (same `mul_hi`-for-`%` idea twice).

## 4. Correctness oracle — π(10ⁿ)

Every version must agree on the count; `run.py`/`scale.py` cross-check this. Known
values (use these on the new machine to validate the CUDA port):

| n | π(10ⁿ) | n | π(10ⁿ) |
|---|---|---|---|
| 3 | 168 | 8 | 5,761,455 |
| 4 | 1,229 | 9 | 50,847,534 |
| 5 | 9,592 | 10 | 455,052,511 |
| 6 | 78,498 | 11 | 4,118,054,813 |
| 7 | 664,579 | 12 | 37,607,912,018 |

## 5. Reference results (Apple M3 Max — to compare against)

- Trial division, N=10⁹: `openmp` ~38 s; `opencl` 736 s (uint64) → 49.8 s (uint32).
- Sieve, N=10¹²: `sieve_cpu` 97.7 s · `sieve_gpu` 196.9 s · **`sieve_gpu_barrett` 56.5 s**.
- Scaling: `results_m3max/scale_sieve_barrett_3-12.*` (Barrett wins from ~10⁹ up, ~2× at 10¹¹–10¹²).

---

## 6. THE NEXT STEP — Windows 11 + CUDA port

**Goal:** build & run the same versions on Windows with an NVIDIA RTX GPU using
**CUDA** as the GPU framework, collect results into a new `results_<machine>/`,
and compare against `results_m3max/`.

**Toolchain to install on the Windows box:**
- **CUDA Toolkit** + a matching **Visual Studio** (MSVC `cl.exe`, which `nvcc` drives).
- **Python 3** (+ `matplotlib`) for the runners/plots.
- Optional: LLVM/clang or `nvc++` if you want real `omp_target` GPU offload (on
  NVIDIA it actually offloads, unlike on Apple where it fell back to the CPU).

**Recommended design:** keep OpenCL for macOS, **add** CUDA `.cu` files for Windows,
and emit the **same binary names** (`bin/sieve_gpu`, etc.) from whichever source the
platform selects — so `results_m3max/` (OpenCL) and `results_rtx/` (CUDA) carry
identical version labels and plot/compare directly.

**Work plan (rough order):**

A. **Build system.** Cleanest is a `CMakeLists.txt` that branches: Apple →
   OpenCL framework (current `.cl` + `opencl`/`sieve_gpu*` host); Windows →
   `enable_language(CUDA)` + the `.cu` ports. Alternative: keep the Unix `Makefile`
   and add a `build.bat`/CMake for Windows only. The CPU/OpenMP/Python parts are
   shared.

B. **CPU versions** (`seq`, `partition`, `stripe`, `atomic_counter`,
   `atomic_dynamic`, `sieve_cpu`): just compile with MSVC/clang. They are pure
   C++17 `std::thread`/`atomic`/`chrono`. Drop the `-pthread` flag (not needed on
   MSVC). `common/*.hpp` are portable.

C. **GPU → CUDA** (`opencl.cpp`→`opencl.cu`, `sieve_gpu.cpp`→`sieve_gpu.cu`,
   `sieve_gpu_barrett.cpp`→`sieve_gpu_barrett.cu`). The kernels in
   `src/prime_kernel.cl` / `src/sieve_kernel.cl` translate almost line-for-line:

   | OpenCL | CUDA |
   |---|---|
   | `__kernel void f(...)` | `__global__ void f(...)` |
   | `get_global_id(0)` | `blockIdx.x*blockDim.x + threadIdx.x` |
   | `get_group_id(0)` / `get_local_id(0)` | `blockIdx.x` / `threadIdx.x` |
   | `get_num_groups(0)` / `get_local_size(0)` | `gridDim.x` / `blockDim.x` |
   | `__local uchar* buf` (sized at launch) | `extern __shared__ unsigned char buf[];` + `f<<<grid,block,shmemBytes>>>` |
   | `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` |
   | `mul_hi(a,b)` (Barrett core) | `__umul64hi(a,b)` |

   CUDA host code (`cudaMalloc`/`cudaMemcpy`/`f<<<grid,block,shmem>>>`) is *less*
   boilerplate than OpenCL, and `nvcc` compiles the kernel at build time — so the
   runtime `KERNEL_PATH` file-loading disappears. `common/bench.hpp` includes fine
   into a `.cu`. Keep the JSON output identical so `run.py`/`plot.py`/`scale.py`
   need no changes.

D. **`run.py` tweak.** `discover_binaries()` currently keys off the Unix
   executable bit (`st_mode & 0o111`); on Windows make it look for `bin/*.exe`.
   That's the only tooling change needed.

E. **Validate** with the π(10ⁿ) oracle (§4) — the CUDA code will be untested until
   it runs on the real GPU, so the count cross-check is the safety net. Start small
   (10⁶, 10⁸) before 10⁹⁺.

F. **Collect results** into `results_<machine>/` (e.g. `results_rtx4070/`) via
   `run.py -o results_rtx.../run.csv --plot` and `scale.py -o results_rtx.../scale.csv`.

G. **Document & push:** add the new machine's numbers next to `results_m3max/` in
   the README, note any differences (predictions below), commit, `git push github master`.

**Tuning knobs likely worth revisiting on NVIDIA:** the sieve's `SEG_NUMS` (NVIDIA
`__shared__` is up to ~100–228 KB/SM vs Apple's 32 KB → larger segments → fewer
divisions → the reversal is milder even before Barrett) and `ngrp`/block size for
occupancy.

**Cheaper shortcut (optional, to validate cross-platform fast):** NVIDIA ships an
OpenCL driver, so the *existing* OpenCL code + `.cl` kernels would run on the RTX
with only build/link changes (link the Khronos/NVIDIA OpenCL lib instead of
`-framework OpenCL`). Low effort; CUDA then becomes a "native API / go faster"
follow-up rather than a prerequisite.

**What will likely differ on Intel + RTX (predictions to check):**
- CPU uint32 vs uint64: bigger gap than ARM's ~1× (x86 64-bit `DIV` is slower) —
  maybe ~1.3–2×, still far below the GPU's ~12×.
- The "unified-memory / on-chip-blocking essential" lesson weakens (discrete VRAM,
  ~2× bandwidth, bigger `__shared__`); a plain global-memory GPU sieve may already win.
- `omp_target` *actually offloads* on NVIDIA (was a CPU-fallback mirage on Apple).
- GPU is much faster in absolute terms; crossovers/sweet-spot shift outward.

---

## 7. Git / housekeeping

- Remote `github` → `git@github.com:syssoft-hc/Count-Primes-with-Claude.git`, branch `master`.
- Kept snapshots live in `results_<machine>/` (underscore-named files are tracked;
  `results.*`/`sweep.*`/`scale.*` and hyphen-timestamped files are git-ignored throwaways).
- `bin/`, `__pycache__/`, `.DS_Store`, `.claude/` are git-ignored.
- After the Windows work: update this HANDOFF's status and push so the next person
  (or session) sees current state.
