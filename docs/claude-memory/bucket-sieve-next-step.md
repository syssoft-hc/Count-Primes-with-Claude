---
name: bucket-sieve-next-step
description: RESOLVED via Barrett reduction — the GPU sieve large-N problem is fixed; bucket sieve only needed far past 1e12
metadata:
  type: project
---

In the `copri` parallel prime-counting project (`~/Desktop/2026S/2026S Heterogeneous Computing/CoPri-Claude`), the GPU-sieve-loses-at-large-N problem was the deferred "bucket sieve" next step. **RESOLVED on 2026-06-01 — but with Barrett reduction, not a bucket sieve.**

**The problem:** `sieve_gpu` only beat `sieve_cpu` in a sweet spot ~10⁹–10¹⁰ and reversed above (0.75× at 10¹¹, 0.50× at 10¹²). Cause: each segment did a `start % p` **64-bit division** per base prime; the Apple GPU emulates 64-bit divide and base-prime count grows as √N.

**The fix that shipped:** `sieve_gpu_barrett` (src/sieve_gpu_barrett.cpp + `sieve_count_barrett` in src/sieve_kernel.cl). Barrett reduction replaces `x % p` with `mul_hi`+mul+subtract using a per-prime `mu=floor(2^64/p)` precomputed on the host. Same parallel structure, no division. Result (snapshot `results_m3max/scale_sieve_barrett_3-12.*`, verified to 1e12): crosses 1.0× vs CPU around 1e9 and STAYS ahead — 1.85× at 1e10, 2.03× at 1e11, 1.73× at 1e12 (56s vs CPU 98s vs old GPU 197s). Reversal gone.

**Bucket sieve status:** no longer needed for the tested range; it would only matter far past 1e12 (drop the per-segment per-large-prime visit entirely). Left as an unexplored road, not a planned step. Recurring motif, now closed: **64-bit integer division was the villain** — beaten by trading divide for multiply (same `mul_hi`-for-`%` idea as uint32 on trial division). Related: [[copri-project-overview]].
