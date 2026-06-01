---
name: copri-project-overview
description: What the copri repo is — a didactic parallel prime-counting project
metadata: 
  node_type: memory
  type: project
  originSessionId: 5be64b6d-7901-4412-989c-93d666be5d7b
---

`~/Desktop/2026S/2026S Heterogeneous Computing/CoPri-Claude` (moved here from `~/Desktop/copri` on 2026-06-01) is a didactic project (a git repo, branch `master`) exploring **how to exploit parallelism on a single host** (Apple M3 Max, 16 cores + GPU) by counting primes ≤ N many ways. Owner uses plain trial division on purpose — the focus is parallelism, not fast primality.

Versions share one `is_prime` (template `is_prime_impl<T>` → `is_prime_uint32/64`): `seq`, `partition`, `stripe`, `atomic_counter`, `atomic_dynamic`, `openmp` (libomp, optional), `omp_target` (GPU offload — falls back to CPU on Apple Silicon), `opencl` (GPU). Two sieve versions (`sieve_cpu`, `sieve_gpu`) deliberately break the shared-is_prime rule.

Tooling: `run.py` (per-version runner → CSV, `-w` width, `--plot`), `plot.py` (bar chart), `sweep.py` (thread scaling), `scale.py` (problem-size scaling). Snapshot naming: underscore = tracked (e.g. `results_10e9_u32.*`), hyphen = git-ignored timestamped throwaway.

Big lessons found: dynamic scheduling wins (~11×); the **striping trap** (even stride collides with even/odd numbers); **uint32 vs uint64** swings the GPU ~12× (no native 64-bit integer divide); **algorithm beats parallelism** (sieve ~1000× faster than trial division); the GPU sieve had a **sweet spot** ~10⁹–10¹⁰ that **`sieve_gpu_barrett` fixed** with Barrett reduction (trade divide for multiply → GPU wins to 1e12, ~2×). Recurring motif now closed: 64-bit integer division was the villain, beaten via `mul_hi`. See [[bucket-sieve-next-step]] (resolved). `sieve.py`→`scale.py`. Versions also include `sieve_gpu_barrett`.
