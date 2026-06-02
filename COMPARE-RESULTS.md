# Comparing experiment sets — M3 Max vs RTX 2080 Ti

This file compares the two kept result snapshots in the repo as **experiment
sets**, i.e. the same benchmark suite run on two different hardware/runtime
platforms. The numbers themselves are reproduced from the CSVs in each folder;
the point here is the *cross-platform* reading, not re-deriving any single
machine's story (for that, see [`A-Random-Walk-to-a-Trillion.md`](A-Random-Walk-to-a-Trillion.md)).

| Set | Folder | Machine | CPU | GPU | GPU runtime |
|---|---|---|---|---|---|
| **A** | [`results_m3max/`](results_m3max/) | Apple M3 Max (macOS) | M3 Max, 16 cores (12P+4E) | integrated Apple GPU | **OpenCL / Metal** |
| **B** | [`results_rtx2080ti/`](results_rtx2080ti/) | Windows box | Intel i9-7900X, 10c/20t | **NVIDIA RTX 2080 Ti** (discrete) | **CUDA** |

## ⚠️ Read this before comparing any GPU row

`opencl`, `sieve_gpu`, and `sieve_gpu_barrett` are **kept labels meaning "the GPU
version,"** not a guarantee of the backend. On the **M3 Max** they are genuine
**OpenCL** (`src/opencl.cpp`, `src/*_kernel.cl`); on the **RTX 2080 Ti** they are
a **CUDA** reimplementation of the same algorithm (`src/opencl.cu`, `src/*.cuh`)
that keeps the old name only so the version columns line up for plotting.

**Consequence:** every M3-vs-RTX **GPU** comparison below moves *two* variables at
once — the **silicon** *and* the **API/runtime** (Apple OpenCL vs native CUDA). A
gap on a GPU row is *device + backend combined*, never the hardware alone. The
only strictly like-for-like cross-machine numbers are the **CPU** rows (`seq`,
`partition`, `sieve_cpu`, …), which are the same C++17 on both. Keep this in mind
throughout; it is the single biggest interpretation hazard in this comparison.

## Coverage — the two sets are not symmetric

| Experiment | Set A (M3 Max) | Set B (RTX 2080 Ti) |
|---|---|---|
| Full version board @ N=10⁸ | ✅ `copri_10e8` (incl. `openmp`, no sieve rows) | ✅ `run_10e8` (incl. sieve rows, no `openmp`) |
| Thread sweep | ✅ `sweep_10e8` (N=10⁸) | ✅ `sweep_10e7` (N=10⁷) |
| Sieve CPU-vs-GPU scaling | ✅ `scale_sieve_3-12` (→10¹²) + `scale_sieve_barrett_3-12` | ✅ `scale_sieve_3-11` (→10¹¹, all three versions in one run) |
| uint32 vs uint64 swing | ✅ several (`results_10e7_both`, `results_10e9_u32`, …) | ❌ only width=32 captured |
| Reaches N = 10¹² (trillion) | ✅ | ❌ (stops at 10¹¹) |

`openmp`/`omp_target` are not built on Windows, so they appear only in Set A.
Sieve numbers below use each platform's run where **all three** sieve versions
were measured together (`scale_sieve_barrett_3-12` for A, `scale_sieve_3-11` for
B) so the within-set comparison is fair.

---

## A. Trial division @ N=10⁸ — the CPU box is the M3 Max

`best_ms`, from `copri_10e8.csv` (A) and `run_10e8.csv` (B). Thread count is
`auto` on each machine, so the **parallel** rows are not core-count-matched; the
**`seq`** row is the clean single-core comparison.

| Version | A: M3 Max | B: RTX box | Faster | Notes |
|---|---:|---:|---|---|
| `seq` (1 core) | 16,277 | 47,799 | **A by 2.9×** | Apple per-core throughput |
| `partition` | 1,745 | 4,745 | A by 2.7× | |
| `atomic_dynamic` | 1,444 | 3,850 | A by 2.7× | best CPU scheduler both sides |
| `opencl` (GPU TD) | 23,151 | 691 | **B by 33.5×** | ⚠️ **OpenCL vs CUDA** — see caveat |

Two opposite stories on one board:

- **CPU: Set A wins everything, even single-threaded** (2.9× on `seq`). The RTX
  box's i9 is the weaker half of that machine.
- **GPU trial division: Set B wins by 33×** — but this is the caveat in the flesh.
  On the M3 Max, OpenCL trial division (23,151 ms) is **slower than one CPU core**
  (0.7×); on the RTX, CUDA trial division hits **69× over `seq`**. The 33× gap
  blends "discrete 2080 Ti ≫ integrated Apple GPU" with "native CUDA ≫ Apple's
  deprecated OpenCL." It is **not** one portable kernel behaving differently.

---

## B. CPU sieve scaling — Set A leads, by a shrinking margin

`sieve_cpu` `best_ms`. Same C++ both sides → **like-for-like**.

| N | A: M3 Max | B: RTX box | A advantage |
|---:|---:|---:|---:|
| 10³ | 0.158 | 3.305 | **20.9×** |
| 10⁷ | 0.643 | 3.904 | 6.1× |
| 10⁸ | 3.953 | 11.781 | 3.0× |
| 10⁹ | 34.6 | 90.1 | 2.6× |
| 10¹⁰ | 397.7 | 973.8 | 2.4× |
| 10¹¹ | 5,556 | 12,479 | 2.2× |

Two regimes:

- **Small N is dominated by fixed overhead.** The RTX box's `sieve_cpu` has a
  ~3 ms floor (it never drops below 3.2 ms even at N=10³), while the M3 Max sits
  at ~0.15 ms — a 20× gap that is **startup cost, not compute**.
- **Large N is memory-bandwidth bound,** and the gap converges to a steady
  **~2.2–2.4×** in Apple's favor — its unified-memory bandwidth and cache carry
  the segmented sieve.

---

## C. GPU sieve scaling — the discrete card wins, most at small N

`sieve_gpu_barrett` `best_ms`. ⚠️ **CUDA (B) vs OpenCL (A)** — device + backend.

| N | A: M3 Max | B: RTX 2080 Ti | B advantage |
|---:|---:|---:|---:|
| 10³ | 4.75 | 0.301 | **15.8×** |
| 10⁷ | 5.92 | 0.762 | 7.8× |
| 10⁸ | 12.31 | 3.065 | 4.0× |
| 10⁹ | 29.7 | 19.2 | 1.6× |
| 10¹⁰ | 214.8 | 136.0 | 1.6× |
| 10¹¹ | 2,732 | 1,699 | 1.6× |

Mirror image of the CPU sieve:

- **Small-N gap is launch overhead.** CUDA dispatches a kernel in ~0.3 ms; the
  Apple OpenCL path has a ~4.8 ms floor. That alone is most of the 15.8× at N=10³.
  On the RTX, the GPU sieve beats *its own CPU sieve at every N down to 10³*
  (0.30 vs 3.31 ms); on the M3 Max the GPU sieve doesn't overtake the CPU sieve
  until ~10⁹.
- **Large-N gap is raw throughput,** and it settles at a steady **~1.6×** — the
  honest "bigger discrete GPU" margin, once overhead and backend noise wash out.

---

## D. The headline: Barrett reduction is platform-relative

Same optimization (replace the per-segment 64-bit modulo with a multiply+shift),
opposite importance. `sieve_gpu → sieve_gpu_barrett`, `best_ms`.

| N | A: plain → Barrett | A speedup | B: plain → Barrett | B speedup |
|---:|---:|---:|---:|---:|
| 10⁸ | 13.76 → 12.31 | 1.12× | 3.14 → 3.07 | 1.02× |
| 10⁹ | 38.98 → 29.70 | 1.31× | 20.03 → 19.19 | 1.04× |
| 10¹⁰ | 357.7 → 214.8 | 1.67× | 148.1 → 136.0 | 1.09× |
| 10¹¹ | 7,485 → 2,732 | **2.74×** | 1,983 → 1,699 | 1.17× |
| 10¹² | 196,919 → 56,467 | **3.49×** | — (not run) | — |

**On Apple, Barrett is essential; on NVIDIA, it's a nice-to-have.** Up to **3.5×**
on the M3 Max at 10¹², but never more than **1.17×** on the RTX. The cause is
hardware: the Apple GPU has no fast native 64-bit integer divide, so modulo is the
bottleneck and removing it pays enormously; the RTX 2080 Ti divides in hardware,
so the sieve there is memory-bound and Barrett barely moves it.

**Barrett even changes *which processor wins* on Apple.** Without it, the M3 Max
GPU sieve is **slower than its own CPU sieve** at large N:

| N | A: CPU sieve | A: GPU plain | A: GPU Barrett |
|---:|---:|---:|---:|
| 10¹¹ | 5,556 | 7,485 ❌ slower than CPU | **2,732** ✅ |
| 10¹² | 97,744 | 196,919 ❌ slower than CPU | **56,467** ✅ |

So on Apple, Barrett is what *rescues the GPU sieve* into a win. On the RTX, the
GPU sieve already beats the CPU at every N — the rescue was never needed. This is
the whole saga in one line: **the villain (the missing 64-bit divide) lives on one
GPU, not on "GPUs."**

---

## E. What's invariant across platforms — thread scaling

Thread sweeps (`sweep_10e8.csv` for A at N=10⁸, `sweep_10e7.csv` for B at N=10⁷;
different N, but the *efficiency shape* is the comparable thing). All CPU, so
like-for-like.

- **The striping trap reproduces exactly.** `stripe` and `atomic_counter` collapse
  to ~0.5 parallel efficiency on even thread counts on *both* machines (e.g. eff at
  2 threads: A 0.489, B 0.498; at 12 threads: A 0.30, B 0.31). This is a property
  of the *decomposition* (the stride-2 interaction with even thread counts), not
  the silicon — so it travels.
- **`atomic_dynamic` is the robust scheduler on both.** Work-stealing holds the
  best efficiency at 16 threads on each box (A 0.708, B 0.696).

**Lessons travel; numbers don't.** The *ranking* of approaches and the scaling
pathologies are identical across platforms; the absolute milliseconds are
hardware-specific, which is exactly why each machine gets its own re-measured
`results_<machine>/` folder.

---

## Bottom line

1. **Portable code ≠ portable performance.** The GPU-trial-division version swings
   from villain (0.7× on Apple OpenCL) to hero (69× on NVIDIA CUDA) — but that
   comparison changes *both* device and runtime, so it overstates the hardware gap.
2. **Right tool per machine.** Set A (M3 Max) is a CPU monster — it wins `seq` and
   the CPU sieve outright, and its GPU only helps at huge N *with Barrett*. Set B
   (RTX box) pairs a weaker CPU with a GPU that wins the sieve at any size.
3. **The clean, backend-free numbers** are the CPU rows: M3 Max leads ~2.2–2.9× and
   the discrete GPU's honest large-N sieve margin is ~1.6×.
4. **Optimizations are platform-relative.** Barrett is *essential* on Apple (up to
   3.5×, and it's what makes the GPU sieve beat the CPU at all) and *marginal* on
   NVIDIA (≤1.17×).

## Provenance

- Set A: `results_m3max/{copri_10e8,sweep_10e8,scale_sieve_3-12,scale_sieve_barrett_3-12,results_sieve_10e9}.csv`
- Set B: `results_rtx2080ti/{run_10e8,sweep_10e7,scale_sieve_3-11}.csv`

All times are `best_ms` (minimum over repeats) unless noted. CPU rows are the same
binary on both platforms; GPU rows are OpenCL on Set A and CUDA on Set B (see the
caveat above). Re-running on a new machine should drop a `results_<machine>/`
folder and extend the tables here rather than overwrite either set.
