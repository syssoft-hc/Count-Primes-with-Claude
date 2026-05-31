# copri — counting primes in parallel

A didactic project: count the primes in `[1, N]` many different ways, all using
the **same** trivial primality test, to study **how to exploit parallelism on a
single host** (here: an Apple M3 Max, 16 cores + an integrated GPU).

The goal is *not* fast primality testing. Every version uses the same naive
trial division (`is_prime` in [`common/prime.hpp`](common/prime.hpp), mirrored
in OpenCL C in [`src/prime_kernel.cl`](src/prime_kernel.cl)):

```cpp
bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (uint64_t c = 3; c * c <= n; c += 2)   // odd divisors up to sqrt(n)
        if (n % c == 0) return false;
    return true;
}
```

Because `is_prime(n)` costs ~`sqrt(n)`, large candidates are far more expensive
than small ones — which is exactly what makes **load balancing** interesting.

## Versions

| Binary | Technique | What it teaches |
|---|---|---|
| `seq` | sequential baseline | reference for correctness & speedup |
| `partition` | static **block** split across threads | simple, but **load-imbalanced** (top block is the slowest) |
| `stripe` | **cyclic** split (`n = t, t+P, t+2P, …`) | interleaving *usually* balances the `sqrt(n)` cost — but watch what stride 2 does (see Thread scaling) |
| `atomic_counter` | one shared `std::atomic` counter | **contention**: an atomic per *prime* is the wrong granularity |
| `atomic_dynamic` | atomic **work-stealing** cursor | dynamic scheduling: atomic per *chunk* → balance *and* low contention |
| `openmp` | `#pragma omp parallel for` (optional) | the high-level equivalent of the hand-rolled versions |
| `opencl` | GPU, striped grid + host reduction | when offloading to the GPU **does and doesn't** pay off |

All binaries share one CLI and output format:

```
./bin/<version> <N> [threads]
```

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

- **OpenMP** is optional. Apple clang has no bundled `omp.h`; the Makefile builds
  `openmp` only if it finds a libomp install. To enable it: `brew install libomp`,
  then `make`. Otherwise it is silently skipped and `run.py` ignores it.
- **OpenCL** uses the macOS framework (deprecated but functional). On Linux,
  install an OpenCL ICD and adjust the link flags in the Makefile.

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
```

`run.py` builds (`make`), runs each version, **verifies every version agrees on
the prime count**, prints a table, and writes a timestamped CSV
(`results-<YYYYmmdd-HHMMSS>.csv`) so a long run never overwrites an earlier one.
Pass `-o name.csv` to choose the path explicitly. Example contents:

```
version,N,threads,repeats,count,best_ms,median_ms,speedup_vs_seq
seq,10000000,auto,2,664579,628.055,636.984,1.0
partition,10000000,auto,2,664579,67.398,67.985,9.32
...
```

## Plotting

`run.py --plot` charts the run it just produced. To (re)chart an existing CSV:

```sh
python3 plot.py                       # results.csv -> results.png
python3 plot.py -i copri_10e8.csv -o copri_10e8.png
python3 plot.py --show                # also open an interactive window
```

Two panels share the version axis: best runtime (log scale) and speedup vs
`seq`; the fastest version is highlighted. A saved snapshot of the 10⁸ run lives
in `copri_10e8.csv` / `copri_10e8.png`.

## Output files & naming convention

Result files can take a long time to produce, so the runners never overwrite
each other, and the repo distinguishes throwaway runs from snapshots worth
keeping by a single character — **hyphen vs. underscore**:

| Form | Example | Produced by | Git |
|---|---|---|---|
| **hyphen** = auto-timestamped run | `results-20260531-172143.csv`, `sweep-20260531-172143.png` | the **default** output of `run.py` / `sweep.py` (`results-<YYYYmmdd-HHMMSS>` / `sweep-<…>`); the `.png` inherits the stamp | **ignored** (`*-*.csv/png`) |
| **underscore** = named snapshot | `copri_10e8.csv`, `sweep_10e8.png` | you, deliberately — via `-o name.csv` or by renaming a timestamped file | **tracked** |

So every run is preserved on disk under its own timestamp, git stays free of
benchmark clutter, and promoting a run to a permanent, tracked snapshot is just
a rename with an underscore (or `-o sweep_<label>.csv`). Passing an explicit
`-o` always uses that exact path — no timestamp is added.

## Sample results (Apple M3 Max, N = 10⁷)

```
version              best_ms   median_ms   speedup
--------------------------------------------------
seq                  628.055     636.984      1.0x
partition             67.398      67.985     9.32x
stripe                84.921      85.157      7.4x
atomic_counter        93.455      93.581     6.72x
atomic_dynamic        55.167      55.240    11.38x   <- best
opencl               707.384     713.371     0.89x   <- slower than 1 CPU core!
```

Things to notice and discuss:

- **`atomic_dynamic` wins.** Dynamic chunk-stealing keeps all 16 cores busy to
  the very end, while static `partition`/`stripe` leave some cores idle once
  they finish their share (M3 Max also mixes faster *performance* cores with
  slower *efficiency* cores, which amplifies static imbalance).
- **`atomic_counter` is slower than `stripe`** even though they do identical
  work — the only difference is a contended atomic increment *per prime* instead
  of a private accumulator reduced once. Synchronization granularity matters.
- **The GPU loses here**, and that is the point: trial division is dominated by
  64-bit integer `%` (division), which GPUs execute very slowly, and there is no
  data parallelism to amortize it. GPUs shine on wide float/SIMD work — not this.
  Swapping in a sieve, or 32-bit candidates, would change the story.

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
files are git-ignored; rename one (e.g. `sweep_10e8.csv`, underscore) to keep it
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

## Layout

```
common/prime.hpp        shared is_prime() — the ONE primality test
common/bench.hpp        arg parsing, timing, uniform JSON/stderr output
src/*.cpp               one file per approach
src/prime_kernel.cl     OpenCL mirror of is_prime()
Makefile                auto-detects OpenMP / OpenCL
run.py                  per-version runner → results.csv (+ --plot)
plot.py                 bar chart of a results.csv
sweep.py                thread-scaling sweep → sweep.csv + sweep.png
```
