# Test platform — Apple M3 Max (`results_m3max/`)

Hardware and software characteristics of the machine on which every snapshot in
this folder was measured. Captured 2026-06-02 (repo at commit `0dd0f9b`). The
companion set is [`../results_rtx2080ti/`](../results_rtx2080ti/); for the
cross-platform reading see [`../COMPARE-RESULTS.md`](../COMPARE-RESULTS.md).

## Machine

| | |
|---|---|
| Model | MacBook Pro (`Mac15,9`) |
| SoC | Apple **M3 Max** |
| Architecture | arm64 (Apple Silicon) |

## CPU

| | |
|---|---|
| Total cores | **16** — 12 Performance + 4 Efficiency |
| Logical CPUs | 16 (no SMT) |
| Performance core caches | L1i 192 KB · L1d 128 KB · L2 16 MB (per P-core cluster) |
| Efficiency core caches | L1i 128 KB · L1d 64 KB · L2 4 MB (per E-core cluster) |

The 12P/4E split is why the thread-scaling sweeps show a knee: parallel
efficiency holds well up to ~12 threads (the performance cores), then bends as the
slower efficiency cores join. See `sweep_10e8.*`.

## GPU

| | |
|---|---|
| GPU | Apple **M3 Max** (integrated) |
| GPU cores | **40** |
| Vendor | Apple (`0x106b`) |
| Graphics/compute API | Metal 4; OpenCL via the macOS framework |
| 64-bit integer divide | **no fast native instruction** |

That last row is the protagonist of the whole project: the Apple GPU has no fast
hardware 64-bit integer divide, so the sieve's per-segment modulo is the
bottleneck — which is exactly what Barrett reduction (`sieve_gpu_barrett`) routes
around. On the RTX 2080 Ti, which *does* divide in hardware, the same optimization
barely matters (see `../COMPARE-RESULTS.md`, section D).

## Memory

| | |
|---|---|
| Installed | **64 GB** unified (68,719,476,736 bytes) |
| Type | Unified memory (CPU + GPU share one pool) |
| Page size | 16 KB (16,384 bytes) |

Unified memory matters here: the GPU sieve only starts winning once its working
set is blocked into on-chip memory instead of streaming RAM.

## Operating system

| | |
|---|---|
| macOS | **26.5** (build `25F71`) |
| Kernel | Darwin **25.5.0**, arm64 |

## Toolchain & libraries (as the benchmarks were built)

| Component | Version | Used for |
|---|---|---|
| Apple clang | **21.0.0** (`clang-2100.1.1.101`), target `arm64-apple-darwin25.5.0` | all CPU versions (C++17) |
| Command Line Tools | 26.5.0.0.1777544298 | SDK, headers, linker |
| libomp (Homebrew) | **22.1.6** | `openmp` / `omp_target` |
| OpenCL | macOS **`OpenCL.framework`** (Apple's implementation — **deprecated**, OpenCL 1.2-class on Apple Silicon) | `opencl`, `sieve_gpu`, `sieve_gpu_barrett` |
| GNU Make | 3.81 | build (`Makefile`) |
| CMake | 4.3.1 | cross-platform build (not used for this set) |
| Python | **3.11.8** | `run.py` / `sweep.py` / `scale.py` / `plot.py` |
| matplotlib | 3.10.0 | charts (`.png`) |
| numpy | 2.2.5 | charts |

### Build flags

- **CPU:** `clang++ -std=c++17 -O3 -Wall -Wextra -pthread`
- **OpenMP:** as above plus `-Xpreprocessor -fopenmp -I<libomp>/include -L<libomp>/lib -lomp`
- **OpenCL:** as above plus `-framework OpenCL` (kernels loaded at runtime from
  `src/prime_kernel.cl` / `src/sieve_kernel.cl` via `-DKERNEL_PATH=…`)

## Caveat for cross-machine comparison

On this Mac the GPU versions (`opencl`, `sieve_gpu`, `sieve_gpu_barrett`) are
genuine **OpenCL**. On `../results_rtx2080ti/` the same-named binaries are a
**CUDA** reimplementation. So any GPU row compared across the two folders varies
*device and runtime together* — only the CPU rows are strictly like-for-like. See
[`../COMPARE-RESULTS.md`](../COMPARE-RESULTS.md) and the README's
[Windows build section](../README.md#windows-cmake--msvc-gpu-via-cuda).
