# copri — counting primes in parallel

A didactic project: count the primes in `[1, N]` many different ways, all using
the **same** trivial primality test, to study **how to exploit parallelism on a
single host** (here: an Apple M3 Max, 16 cores + an integrated GPU).

> 📖 **Want the story instead of the manual?** [**A Random Walk to a
> Trillion**](A-Random-Walk-to-a-Trillion.md) is a travelogue of how these
> results were found — the wrong turns, the 736-second wall, the recurring
> villain (64-bit integer division), and the GPU's sweet spot — told in the order
> it actually happened.
>
> 🧭 **Resuming on another machine (e.g. Windows + CUDA)?** Read
> [`HANDOFF.md`](HANDOFF.md) first — it's the portable project memory and the
> step-by-step plan for the CUDA port (Claude's `~/.claude` memory doesn't travel
> with the repo).

The goal is *not* fast primality testing. Every version uses the same naive
trial division (one template `is_prime_impl<T>` in
[`common/prime.hpp`](common/prime.hpp), mirrored in OpenCL C in
[`src/prime_kernel.cl`](src/prime_kernel.cl)), exposed at two fixed widths:

```cpp
template <typename T>
bool is_prime_impl(T n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (T c = 3; c * c <= n; c += 2)          // odd divisors up to sqrt(n)
        if (n % c == 0) return false;
    return true;
}
bool is_prime_uint32(uint32_t n) { return is_prime_impl<uint32_t>(n); }
bool is_prime_uint64(uint64_t n) { return is_prime_impl<uint64_t>(n); }
```

Because `is_prime(n)` costs ~`sqrt(n)`, large candidates are far more expensive
than small ones — which is exactly what makes **load balancing** interesting.
The 32- vs 64-bit split matters enormously **on the GPU** — see
[uint32 vs uint64](#uint32-vs-uint64).

## Results at a glance

Counting the primes ≤ 10⁹ (π = 50,847,534) on an Apple M3 Max — the same problem,
five ways:

| version | approach | time | vs `openmp` |
|---|---|---|---|
| `openmp` | best parallel **trial division**, CPU (16 cores) | ~38 s | 1× |
| `opencl` *(uint64)* | trial division, **GPU** | ~736 s | **0.05×** |
| `opencl` *(uint32)* | trial division, GPU | ~50 s | 0.76× |
| `sieve_cpu` | **segmented sieve**, CPU | ~0.05 s | **~730×** |
| `sieve_gpu` | segmented sieve, GPU | ~0.04 s | **~910×** |

Two effects dwarf the ~11× that 16 cores buy over one:

- **Integer width swings the GPU ~12×** — the Apple GPU has no native 64-bit
  integer divide, so uint64 trial division is catastrophic but uint32 is fine
  ([details](#uint32-vs-uint64)).
- **Algorithm swings *everything* ~1000×** — a Sieve of Eratosthenes has no
  division in its inner loop, and on-chip blocking even lets the GPU win
  ([details](#sieve--when-the-gpu-finally-wins)). At 10¹⁰ the gap reaches ~3000×
  (17.5 min vs 0.35 s).

The rest of this README is how each of those numbers comes about, and the
parallelization patterns (partitioning, striping, atomics, dynamic scheduling,
OpenMP, GPU offload) compared along the way.

## Versions

| Binary | Technique | What it teaches |
|---|---|---|
| `seq` | sequential baseline | reference for correctness & speedup |
| `partition` | static **block** split across threads | simple, but **load-imbalanced** (top block is the slowest) |
| `stripe` | **cyclic** split (`n = t, t+P, t+2P, …`) | interleaving *usually* balances the `sqrt(n)` cost — but watch what stride 2 does (see Thread scaling) |
| `atomic_counter` | one shared `std::atomic` counter | **contention**: an atomic per *prime* is the wrong granularity |
| `atomic_dynamic` | atomic **work-stealing** cursor | dynamic scheduling: atomic per *chunk* → balance *and* low contention |
| `openmp` | `#pragma omp parallel for` (optional) | the high-level equivalent of the hand-rolled versions |
| `omp_target` | `#pragma omp target` GPU **offload** (optional) | the offload programming model — and why it falls back to the CPU here (see note) |
| `opencl` | GPU, striped grid + host reduction | when offloading to the GPU **does and doesn't** pay off |
| `sieve_cpu` | parallel **segmented sieve** (std::thread) † | algorithm beats parallelism — ~1000× faster than trial division |
| `sieve_gpu` | segmented sieve in GPU `__local` memory † | the right algorithm is what finally makes the **GPU win** |
| `sieve_gpu_barrett` | GPU sieve with **Barrett reduction** † | trade the per-segment 64-bit division for multiplies → GPU keeps winning at large N |

† `sieve_cpu` / `sieve_gpu` / `sieve_gpu_barrett` deliberately **break the shared-`is_prime()` rule** —
a sieve marks composites instead of testing each candidate. They are the answer
to "could the GPU ever win here?" (yes — see [Sieve](#sieve--when-the-gpu-finally-wins)).

All binaries share one CLI and output format:

```
./bin/<version> <N> [threads] [u32|u64|auto]
```

The optional width token (`auto` is the default) picks the integer type for
`is_prime`: `auto` uses u32 when `N` ≤ 4×10⁹ and u64 above it. The tokens may
appear in any order (a bare number is the thread count).

They print one JSON line to **stdout** (consumed by `run.py`) and a human
summary to **stderr**:

```
[atomic_dynamic] N=10000000 threads=16 -> 664579 primes in 55.17 ms
```

## Build

```sh
make            # builds every available version into bin/
make clean
```

- **OpenMP** (`openmp` and `omp_target`) is optional. Apple clang has no bundled
  `omp.h`; the Makefile builds these only if it finds a libomp install. To enable
  them: `brew install libomp`, then `make`. Otherwise they are silently skipped
  and `run.py` ignores them.
- **OpenCL** uses the macOS framework (deprecated but functional). On Linux,
  install an OpenCL ICD and adjust the link flags in the Makefile.

### Windows (CMake + MSVC, GPU via CUDA)

The Unix `Makefile` is macOS/Linux only. On Windows an **additive**
`CMakeLists.txt` builds the same sources (the macOS path is untouched). It needs
**Visual Studio Build Tools** (MSVC + the C++ workload) and, for the GPU
versions, the **CUDA Toolkit** (which also supplies the OpenCL headers/lib).

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Binaries land in `bin\<name>.exe`. You usually don't even call cmake yourself:
`run.py` locates MSVC + cmake and configures/builds for you, so from any shell

```bat
python run.py -n 10000000
```

just works. The GPU versions (`opencl`, `sieve_gpu`, `sieve_gpu_barrett`) are
built from **CUDA** `.cu` sources on Windows and from **OpenCL** on macOS, under
the *same* binary names, so results compare directly across machines. Choose the
backend with `-DCOPRI_GPU_BACKEND=cuda|opencl|off` (default: CUDA when a CUDA
compiler is found, else OpenCL). `openmp`/`omp_target` are not built on Windows.

### `omp_target` and GPU offload — read this before trusting the number

`omp_target` uses `#pragma omp target teams distribute parallel for`, the OpenMP
way to offload a loop to a GPU. **On Apple Silicon it does not actually use the
GPU.** OpenMP offloads only to NVPTX (NVIDIA), AMDGPU, or SPIR-V devices; the
Apple GPU is reachable only via Metal/OpenCL, and no offload-capable LLVM ships
for it. So here `omp_get_num_devices() == 0` and the `target` region transparently
**falls back to the host CPU** — which is why its time is close to `openmp`, not
to a GPU. The binary prints which path it took:

```
[omp_target] offload devices=0 -> target runs on HOST (CPU fallback)
```

The *same source* offloads to a real GPU on a machine with an NVIDIA/AMD card and
an offload-enabled build, e.g.
`clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda …`. It is included to
teach the offload programming model and the portability/fallback story; for the
real GPU path on this Mac, use `opencl`. (For that reason `omp_target` is left out
of the `sweep.py` thread sweep, which is about CPU thread scaling.)

The shared `is_prime()` stays a single implementation: `omp_target.cpp` includes
`common/prime.hpp` inside a `#pragma omp declare target` region so the compiler
also emits a device-side copy, rather than duplicating the function.

## Run the benchmark

```sh
python3 run.py                  # N=1_000_000, all built versions, 3 repeats
python3 run.py -n 10000000      # larger limit
python3 run.py -n 5000000 -r 5  # more repeats (best & median recorded)
python3 run.py -t 8             # pin CPU versions to 8 threads
python3 run.py --only seq stripe
python3 run.py --no-build       # use whatever is already in bin/
python3 run.py --plot           # also render a bar chart (results.png)
python3 run.py --plot chart.png # ... to a chosen path
python3 run.py -w 32            # force uint32 (auto/32/64/both)
python3 run.py -w both          # run every version at u32 AND u64
```

`run.py` builds (`make`), runs each version, **verifies every version agrees on
the prime count**, prints a table, and writes a timestamped CSV
(`results-<YYYYmmdd-HHMMSS>.csv`) so a long run never overwrites an earlier one.
Pass `-o name.csv` to choose the path explicitly. Example contents:

```
version,N,threads,width,repeats,count,best_ms,median_ms,speedup_vs_seq
seq,10000000,auto,64,2,664579,628.055,636.984,1.0
partition,10000000,auto,64,2,664579,67.398,67.985,9.32
...
```

## Plotting

`run.py --plot` charts the run it just produced. To (re)chart an existing CSV:

```sh
python3 plot.py                       # results.csv -> results.png
python3 plot.py -i results_m3max/copri_10e8.csv -o results_m3max/copri_10e8.png
python3 plot.py --show                # also open an interactive window
```

Two panels share the version axis: best runtime (log scale) and speedup vs
`seq`; the fastest version is highlighted. A saved snapshot of the 10⁸ run lives
in `results_m3max/copri_10e8.csv` / `results_m3max/copri_10e8.png`.

## Output files & naming convention

Result files can take a long time to produce, so the runners never overwrite
each other, and the repo distinguishes throwaway runs from snapshots worth
keeping by a single character — **hyphen vs. underscore**:

| Form | Example | Produced by | Git |
|---|---|---|---|
| **hyphen** = auto-timestamped run | `results-20260531-172143.csv`, `sweep-20260531-172143.png` | the **default** output of `run.py` / `sweep.py` (`results-<YYYYmmdd-HHMMSS>` / `sweep-<…>`); the `.png` inherits the stamp | **ignored** (`*-*.csv/png`) |
| **underscore** = named snapshot | `results_m3max/copri_10e8.csv`, `results_m3max/sweep_10e8.png` | you, deliberately — via `-o name.csv` or by renaming a timestamped file | **tracked** |

So every run is preserved on disk under its own timestamp, git stays free of
benchmark clutter, and promoting a run to a permanent, tracked snapshot is just
a rename with an underscore (or `-o sweep_<label>.csv`). Passing an explicit
`-o` always uses that exact path — no timestamp is added.

## Sample results (Apple M3 Max, N = 10⁷, uint32)

`auto` picks uint32 for N = 10⁷, so this is what `python3 run.py -n 10000000`
prints (snapshot: `results_m3max/results_10e7_u32.*`):

```
version              best_ms   median_ms   speedup
--------------------------------------------------
seq                  623.923     625.015      1.0x
partition             66.665      77.106     9.36x
stripe                86.924      86.973     7.18x
atomic_counter        93.493      93.623     6.67x
atomic_dynamic        55.141      55.645    11.32x   <- best trial-division
openmp                55.363      55.451    11.27x
omp_target            68.478      70.002     9.11x   (offload falls back to CPU)
opencl                60.169      62.062    10.37x   (GPU, uint32)
···················· different algorithm: segmented sieve (see "Sieve" below) ····
sieve_cpu              0.554       0.594     1126x   <- fastest overall
sieve_gpu              5.282       5.547      118x   (GPU; loses to sieve_cpu at this small N)
```

Note the scale break: the sieves are ~100–1000× faster than *any* trial-division
version — the algorithm matters far more than the parallelization. At this small
N the CPU sieve also beats the GPU sieve (kernel-launch overhead); the GPU sieve
only pulls ahead at N≈10⁹ (see [Sieve](#sieve--when-the-gpu-finally-wins)).

The gap is most stark at the **N=10¹⁰ capstone** (snapshot `results_m3max/results_10e10.*`):
the best parallel trial-division CPU version takes **~17.5 minutes**, while
`sieve_gpu` counts the same π(10¹⁰)=455,052,511 in **~0.35 s** — a ~**3000×**
algorithm gap. Details in [Sieve](#sieve--when-the-gpu-finally-wins).

Things to notice and discuss:

- **`atomic_dynamic` and `openmp` win.** Dynamic chunk-stealing keeps all 16
  cores busy to the very end, while static `partition`/`stripe` leave some cores
  idle once they finish their share (M3 Max also mixes faster *performance* cores
  with slower *efficiency* cores, which amplifies static imbalance).
- **`atomic_counter` is slower than `stripe`** even though they do identical
  work — the only difference is a contended atomic increment *per prime* instead
  of a private accumulator reduced once. Synchronization granularity matters.
- **The GPU is competitive here (10.37×) — but only in uint32.** The very same
  `opencl` run in uint64 is ~0.89× (slower than one CPU core), an ~12× swing from
  the integer width alone: the Apple GPU has no native 64-bit integer divide. So
  the earlier "GPU loses" result was a *64-bit integer* story, not a GPU limit —
  see [uint32 vs uint64](#uint32-vs-uint64). For N above 4×10⁹ you must use
  uint64, and there the GPU does lose.

## uint32 vs uint64

Every version can run with a 32- or 64-bit `is_prime` (the `auto`/`u32`/`u64`
token, or `run.py -w`). Same algorithm, only the integer width differs. It barely
matters on the CPU but is huge on the GPU:

| | uint64 | uint32 | u32 speedup |
|---|---|---|---|
| **CPU**, 1 thread, N=10⁷ | 622 ms | 624 ms | ~**1.0×** |
| **GPU** (OpenCL), N=10⁷ | 706 ms | 63 ms | ~**11×** |
| **GPU** (OpenCL), N=5×10⁷ | 8169 ms | 631 ms | ~**13×** |
| **GPU** (OpenCL), N=10⁹ | 736 s | 49.8 s | ~**15×** |

Why: Apple's ARM64 CPU cores have a native 64-bit integer divide, so `n % c`
costs about the same either way. The Apple **GPU has no native 64-bit integer
divide** — `ulong % ulong` is emulated from many 32-bit ops — so the hot
instruction gets ~12× cheaper in `uint`. At N=10⁷ that takes OpenCL from ~13×
*slower* than the fastest CPU version to roughly a **tie** with the 16-core CPU.
So most of the "GPU loses" result earlier is really "64-bit integer division is
the wrong tool", not "the GPU is useless".

**At the headline scale (N=10⁹)** the same effect is decisive. Comparing the
fastest CPU version against the GPU (snapshots `results_m3max/results_10e9.*` vs
`results_m3max/results_10e9_u32.*`):

| version | uint64 | uint32 |
|---|---|---|
| `openmp` (fastest CPU) | 38.2 s | 38.7 s |
| `atomic_dynamic` | 39.0 s | 38.9 s |
| `opencl` (GPU) | **736 s** | **49.8 s** |

uint32 makes the GPU ~**14.8×** faster and shrinks its gap to the 16-core CPU
from ~**19×** slower down to ~**1.3×** — same scale, same code, only the integer
width changed.

**The trade-off — correctness headroom.** uint32 is only safe while the
candidate *and* `c*c` stay below 2³². The shared `kU32SafeMax = 4×10⁹` guards
this: `auto` (and `run.py`) use u32 up to 4×10⁹ and switch to u64 above it, and
forcing `u32` on a larger `N` warns and falls back to u64. That is exactly the
">10⁹ in very rare cases" path — it keeps working, just without the GPU's 32-bit
speedup.

```sh
python3 run.py -n 50000000 -w both     # compare every version at u32 vs u64
```

In `-w both` each version is run twice, labelled `<ver>-u32` / `<ver>-u64`, and
each is compared to the sequential baseline *of the same width*.

## Sieve — when the GPU finally wins

Everything above keeps the same trial-division `is_prime`, so the comparisons are
fair — but trial division is a *terrible* algorithm, and no amount of parallelism
fixes that. `sieve_cpu` and `sieve_gpu` break that rule on purpose: they use a
**segmented Sieve of Eratosthenes**, whose inner loop is `mark[j]=1; j+=2p` — pure
additions and byte writes, **no division at all**. Two payoffs:

**1. Algorithm beats parallelism.** `sieve_cpu` counts π(10⁷) in ~0.8 ms; the
trial-division `seq` takes ~624 ms, and even the best parallel trial-division
version ~55 ms. The algorithm change is ~**1000×** — far more than the ~11× the
16 cores buy. *Pick the right algorithm before you parallelize.*

**2. The right algorithm is what lets the GPU win.** At N=10⁹ (snapshot
`results_m3max/results_sieve_10e9.*`):

```
version       best_ms        note
-------------------------------------------------------
openmp        38920    trial division, fastest CPU
opencl        49807    trial division, GPU
sieve_cpu        52    segmented sieve, 16 cores
sieve_gpu        42    segmented sieve, GPU   <- fastest
```

The sieve is ~**1000×** faster than trial division, and now **`sieve_gpu` beats
`sieve_cpu`** (~42 vs ~52 ms; with warm best-of-3 runs ~28 vs ~34 ms). This is
the direct answer to "[would NVIDIA be better / is the Mac GPU weak]" — the GPU
was never the problem, *trial division* was. Give the GPU a regular,
division-free, bandwidth-bound kernel and it pulls ahead.

**The gap only widens with scale.** At N=10¹⁰ (snapshot `results_m3max/results_10e10.*`) the
contrast becomes absurd — the best parallel trial-division CPU version takes
**17.5 minutes**, the sieves under **0.4 s**:

```
version       best_ms        note
-------------------------------------------------------
openmp      1052873    trial division (uint64), fastest CPU  (~17.5 min)
sieve_cpu       396    segmented sieve, 16 cores
sieve_gpu       350    segmented sieve, GPU   <- fastest, 3012x vs openmp
```

That is a ~**3000×** algorithm gap — the same π(10¹⁰)=455,052,511 in 17.5 minutes
or a third of a second, depending only on the algorithm. The GPU sieve keeps its
~1.13× edge over the CPU sieve at 10¹⁰ — but *only within a sweet spot*; push N
higher and it reverses (next section).

Getting there took one real lesson: a **first** GPU sieve that streamed a 500 MB
composite array through global memory *lost* to the CPU (155 ms vs 34 ms), because
on this unified-memory Mac the GPU and CPU share the same RAM — and the CPU sieve
is **cache-blocked**, so it stays on-chip and barely touches it. The fix was to
block the GPU sieve the same way: each work-group sieves one segment in fast
**`__local` (on-chip) memory** instead of global RAM. That on-chip blocking — not
raw FLOPS — is what makes the GPU competitive on a memory-bound task. (`sieve_gpu`
still loses at small N, e.g. 10⁸, where its kernel-launch overhead dominates the
tiny amount of work.)

Both sieves are memory-bound, so the `u32`/`u64` width is irrelevant — they always
report `u64` and ignore `-w`.

### The GPU sieve has a sweet spot, not a monotonic win (`scale.py`)

"Bigger N → GPU wins by more" sounds obvious, and is **wrong**. `scale.py` fixes
the two sieves and varies the *problem size* N = 10ⁿ (snapshot
`results_m3max/scale_sieve_3-12.*`, n = 3…12):

```sh
python3 scale.py --exp 3-12       # plots runtime vs N and gpu/cpu speedup vs N
```

```
       N    sieve_cpu    sieve_gpu   gpu/cpu
--------------------------------------------
10^3        0.16 ms       4.6 ms      0.03x   GPU launch overhead dominates
10^6        0.18 ms       4.4 ms      0.04x
10^8        3.2  ms       7.5 ms      0.42x
10^9        33.4 ms      27.6 ms      1.21x   <- GPU pulls ahead
10^10      412.9 ms     341.0 ms      1.21x   <- peak
10^11      5592  ms     7468  ms      0.75x   <- reverses
10^12     97419  ms    197277  ms      0.49x   CPU ~2x faster
```

The GPU has an **operating window** (~10⁹–10¹⁰), not a permanent edge:

- **Below ~10⁹** it loses to kernel-launch + dispatch overhead (the CPU sieve is
  already sub-millisecond at 10⁶).
- **At ~10⁹–10¹⁰** the segment marking — division-free, bandwidth-bound — dominates,
  and the GPU's memory bandwidth wins (~1.2×).
- **Above ~10¹⁰ it reverses.** Each segment's setup needs a `start % p` —
  a **64-bit division** — for every base prime, and the number of base primes
  grows as √N (≈78 000 at 10¹²). The Apple GPU has **no native 64-bit integer
  divide** (the same wall `opencl` hit), and its `__local` segments are capped at
  32 KB — ~8× smaller than the CPU sieve's 256 KB — so it pays that emulated
  division ~8× more often. Past ~10¹⁰ that setup cost overtakes the bandwidth win.

So the recurring villain — 64-bit integer division — had the last word even for
the sieve… until we took it on directly.

### Beating the wall with Barrett reduction (`sieve_gpu_barrett`)

If the wall is the 64-bit `%`, remove it. **Barrett reduction** computes `x % p`
with two multiplies and a subtract instead of a divide, using a per-prime
constant `μ = ⌊2⁶⁴/p⌋` precomputed once on the host (where division is cheap):

```c
ulong q = mul_hi(x, mu);   // ≈ x / p
ulong r = x - q * p;       // x mod p, off by ≤ ~2p
while (r >= p) r -= p;     // ≤ 2 corrections
```

`sieve_gpu_barrett` is `sieve_gpu` with exactly that swap — same grid-strided,
`__local`-blocked structure, no division. The reversal disappears (snapshot
`results_m3max/scale_sieve_barrett_3-12.*`):

```
        N    sieve_cpu   sieve_gpu   sieve_gpu_barrett   barrett/cpu
------------------------------------------------------------------------
 10^9        34.6 ms      39.0 ms          29.7 ms          1.17x
 10^10      397.7 ms     357.7 ms         214.8 ms          1.85x
 10^11      5556  ms     7485  ms         2732  ms          2.03x
 10^12     97744  ms   196919  ms        56467  ms          1.73x
```

Where the plain GPU sieve fell to 0.50× at 10¹², Barrett *grows* its lead to
~2× and holds it — the GPU now beats the 16-core CPU across the whole large-N
range (it counts π(10¹²)=37,607,912,018 in **56 s vs the CPU's 98 s**). The
honest takeaway updates: a GPU is the right tool inside a window — and you can
*widen that window* by trading its weakness (division) for its strength
(multiplication), the same `mul_hi`-for-`%` trick that uint32 used on trial
division. (A bucket sieve — carrying each prime's offset across segments to drop
the per-segment work entirely — would matter only far past 10¹²; Barrett made it
unnecessary here.)

## Thread scaling (`sweep.py`)

Where `run.py` fixes the thread count and varies the version, `sweep.py` fixes
`N` and varies the **thread count** per version, to show how each approach
*scales* as cores are added:

```sh
python3 sweep.py                        # N=10^7, threads 1,2,4,8,12,16
python3 sweep.py -n 50000000 -p 1,2,4,8,16
python3 sweep.py --versions stripe atomic_dynamic
```

It writes a timestamped `sweep-<YYYYmmdd-HHMMSS>.csv` (one row per version ×
thread count, with `speedup_self` and parallel `efficiency = speedup / threads`)
and a matching `.png`, a two-panel line plot: speedup vs threads (with an ideal
`y = x` line) and efficiency. Pass `-o name.csv` to choose the path. `seq`
(always 1 thread) and `opencl` (GPU work-items, not threads) are excluded.

Both runners timestamp their **default** output filenames, so the (often slow
to produce) result files are never accidentally clobbered. Auto-timestamped
files are git-ignored; rename one (e.g. `results_m3max/sweep_10e8.csv`, underscore) to keep it
as a tracked snapshot.

### The striping trap — cyclic distribution can collide with the data

The sweep exposes a subtle, important failure of `stripe`. With `P` threads,
thread `t` handles `n = 2+t, 2+t+P, 2+t+2P, …` — a stride of `P`. When **`P` is
even, that stride shares the factor 2 with the integers**, so each thread's
residue class is *entirely even or entirely odd*:

- `P = 2`: thread 0 gets `2,4,6,8,…` (all even → `is_prime` returns in O(1)),
  thread 1 gets `3,5,7,9,…` (all odd → full `sqrt(n)` trial division). One thread
  does *all* the real work — **2 threads give ~1× speedup, no gain at all.**
- `P = 4`: 2 of the 4 residue classes are even (cheap), 2 are odd (expensive) →
  ~2× instead of 4×. In general only the ~`P/2` odd-residue threads do real work.

Measured efficiency at `N = 10⁷` makes it vivid:

```
version           1→2 threads   efficiency @ 2t    efficiency @ 16t
atomic_dynamic       1.95×           0.97               0.71
openmp               1.95×           0.98               0.71
partition            1.59×           0.80               0.57
stripe               1.01×           0.50  (!)          0.45
atomic_counter       1.01×           0.50  (!)          0.42
```

`atomic_dynamic` / `openmp` scale near-ideally until the M3 Max's 4 slower
*efficiency* cores join at 12→16 threads (the visible knee). `stripe` and
`atomic_counter` (which is also striped) sit at ~0.5 efficiency on even thread
counts because half their threads are stuck on the trivial even numbers.

**Lessons:** (1) cyclic/striped distribution is only balanced when the stride is
*coprime to any structure in the work* — here it must avoid the factor 2; a
stride of an odd `P`, or skipping evens entirely, fixes it. (2) Dynamic
work-stealing (`atomic_dynamic`, `openmp`) sidesteps the whole problem because it
never bakes the assignment into a fixed pattern.

## Adding a new version

1. Add `src/<name>.cpp`, include `common/prime.hpp` + `common/bench.hpp`, and
   call `run_and_report("<name>", args, lambda_returning_count)`.
2. Add `<name>` to `PORTABLE` in the `Makefile` (or give it its own rule if it
   needs extra flags).
3. `run.py` discovers it automatically; add it to `ORDER` for a fixed position.

## Snapshots

Tracked result sets (each a `.csv` + a `.png`), kept because they were slow to
produce and each isolates one lesson. Underscored names are deliberate snapshots
(timestamped throwaway runs are git-ignored — see the naming convention above).

All of these live under **`results_m3max/`** — the machine they were measured on.
The numbers are hardware-specific, so if you reproduce the experiment on another
machine, collect *its* results into a sibling directory named for it
(`results_<machine>/`, e.g. `results_rtx4090/`) rather than overwriting these.

| Snapshot | N | What it shows |
|---|---|---|
| `results_m3max/copri_10e8` | 10⁸ | baseline comparison of all versions (uint64) — `atomic_dynamic`/`openmp` win, GPU loses |
| `results_m3max/sweep_10e8` | 10⁸ | thread-scaling sweep: speedup & efficiency vs core count, the P/E-core knee, the striping trap |
| `results_m3max/results_10e7_u32` | 10⁷ | all versions in uint32 — the GPU becomes competitive (10.4×) |
| `results_m3max/results_10e7_both` | 10⁷ | every version at u32 **and** u64 side by side — CPU a wash, `opencl` swings ~12× |
| `results_m3max/results_10e9` | 10⁹ | fastest CPU vs GPU, **uint64** — GPU ~19× slower (736 s) |
| `results_m3max/results_10e9_u32` | 10⁹ | same, **uint32** — GPU gap shrinks to ~1.3× (49.8 s) |
| `results_m3max/results_sieve_10e9` | 10⁹ | sieve vs trial division — sieve ~1000× faster, `sieve_gpu` overtakes `sieve_cpu` |
| `results_m3max/results_10e10` | 10¹⁰ | **capstone**: trial-division CPU ~17.5 min vs sieves ~0.35 s (~3000×) |
| `results_m3max/scale_sieve_3-12` | 10³–10¹² | sieve CPU-vs-GPU scaling: the GPU's ~10⁹–10¹⁰ **sweet spot** and the reversal beyond |
| `results_m3max/scale_sieve_barrett_3-12` | 10³–10¹² | adds `sieve_gpu_barrett`: Barrett reduction kills the reversal, GPU lead grows to ~2× at 10¹¹–10¹² |

Re-chart any of them with
`python3 plot.py -i results_m3max/<name>.csv -o results_m3max/<name>.png`.

## Layout

```
common/prime.hpp        shared is_prime() — the ONE primality test
common/bench.hpp        arg parsing, timing, uniform JSON/stderr output
common/sieve_common.hpp base primes for the sieve versions
src/*.cpp               one file per approach (CPU + OpenCL GPU hosts)
src/prime_kernel.cl     OpenCL mirror of is_prime() (u32 + u64 kernels)
src/sieve_kernel.cl     OpenCL segmented sieve (blocked in __local memory)
src/*.cu                CUDA ports of the GPU versions (Windows/NVIDIA)
src/prime_kernel.cuh    CUDA mirror of is_prime() kernels
src/sieve_kernel.cuh    CUDA segmented sieve (blocked in __shared__ memory)
Makefile                macOS/Linux build; auto-detects OpenMP / OpenCL
CMakeLists.txt          cross-platform build (Windows MSVC + CUDA/OpenCL)
run.py                  per-version runner → results.csv (+ --plot, -w width)
plot.py                 bar chart of a results.csv
sweep.py                thread-scaling sweep → sweep.csv + sweep.png
scale.py                problem-size sweep (N=10^n) → scale.csv + scale.png
results_m3max/          kept snapshots for the Apple M3 Max (.csv + .png)
results_rtx2080ti/      kept snapshots for the Windows RTX 2080 Ti (.csv + .png)
```
