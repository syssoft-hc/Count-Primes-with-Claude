// OpenCL kernel: segmented Sieve of Eratosthenes on the GPU, blocked in
// on-chip __local memory (the GPU equivalent of the CPU's cache blocking).
//
// Odd numbers only: within a segment, slot i represents number segLo + 2*i.
// The prime 2 is counted on the host.
//
// One WORK-GROUP cooperatively sieves one segment at a time in fast __local
// memory (never a big global array), then grid-strides to the next segment:
//   1) the group zeroes the local segment buffer,
//   2) each work-item takes a SUBSET of the base primes and marks their odd
//      multiples into the local buffer -- inner loop is `buf[..]=1; m+=2p`,
//      division-free. Different work-items may write the same byte, but always
//      the value 1, so the race is benign (byte stores, no read-modify-write),
//      and no atomics are needed,
//   3) the group counts survivors.
// Keeping the working set in __local memory makes this bandwidth-light, which is
// what lets the GPU compete with a cache-blocked CPU sieve.

#define SEG_NUMS 32768u            // numbers per segment (even)
#define SEG_ODD  (SEG_NUMS / 2u)   // odd slots per segment -> local bytes

__kernel void sieve_count(const ulong N,
                          const uint nprimes,
                          __global const uint* primes,  // odd base primes >= 3, ascending
                          const ulong nseg,             // total segments
                          __global ulong* partial,
                          __local uchar* buf) {         // SEG_ODD bytes
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);
    const ulong ngrp = get_num_groups(0);
    ulong c = 0;

    for (ulong s = get_group_id(0); s < nseg; s += ngrp) {
        const ulong lo = 3 + s * (ulong)SEG_NUMS;   // segment start (odd)
        if (lo > N) break;
        ulong hi = lo + SEG_NUMS;
        if (hi > N + 1) hi = N + 1;
        const ulong odds = (hi - lo + 1) / 2;       // odd slots in use

        for (uint i = lid; i < odds; i += lsz) buf[i] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Each work-item marks a strided subset of the base primes.
        for (uint k = lid; k < nprimes; k += lsz) {
            const ulong p = primes[k];
            const ulong p2 = p * p;
            if (p2 >= hi) continue;                 // (strided k -> continue, not break)
            ulong start = p2 > lo ? p2 : lo;
            const ulong r = start % p;
            if (r) start += (p - r);                // first multiple of p >= start
            if ((start & 1UL) == 0) start += p;     // odd multiples only
            for (ulong m = start; m < hi; m += 2 * p)
                buf[(m - lo) >> 1] = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = lid; i < odds; i += lsz)
            if (!buf[i]) ++c;                       // segment starts at 3, so no "1" to exclude
        barrier(CLK_LOCAL_MEM_FENCE);               // reuse buf next iteration
    }

    partial[get_global_id(0)] = c;
}

// ---------------------------------------------------------------------------
// sieve_count_barrett: identical to sieve_count, but replaces the per-(segment,
// prime) `start % p` -- the GPU's weak spot, since the Apple GPU has no native
// 64-bit integer divide and emulates it -- with BARRETT REDUCTION: two
// multiplies and a subtract, using a per-prime mu = floor(2^64 / p) precomputed
// once on the host (where 64-bit division is cheap). Same "trade division for
// multiplication" trick that rescued trial division via uint32; here it targets
// the cost that grows as sqrt(N) (the base-prime count) and makes the GPU sieve
// lose at large N.
//
//   q = mul_hi(x, mu) ~ floor(x/p);  r = x - q*p in [0, ~3p);  subtract p <= 2x.

inline ulong mod_barrett(ulong x, ulong p, ulong mu) {
    ulong q = mul_hi(x, mu);     // high 64 bits of x*mu ~ floor(x/p)
    ulong r = x - q * p;         // x mod p, possibly off by up to ~2p
    while (r >= p) r -= p;       // normalize (loops at most ~2 times)
    return r;
}

__kernel void sieve_count_barrett(const ulong N,
                                  const uint nprimes,
                                  __global const uint*  primes,
                                  __global const ulong* mus,   // mus[k] = floor(2^64/primes[k])
                                  const ulong nseg,
                                  __global ulong* partial,
                                  __local uchar* buf) {
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);
    const ulong ngrp = get_num_groups(0);
    ulong c = 0;

    for (ulong s = get_group_id(0); s < nseg; s += ngrp) {
        const ulong lo = 3 + s * (ulong)SEG_NUMS;
        if (lo > N) break;
        ulong hi = lo + SEG_NUMS;
        if (hi > N + 1) hi = N + 1;
        const ulong odds = (hi - lo + 1) / 2;

        for (uint i = lid; i < odds; i += lsz) buf[i] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint k = lid; k < nprimes; k += lsz) {
            const ulong p = primes[k];
            const ulong p2 = p * p;
            if (p2 >= hi) continue;
            ulong start = p2 > lo ? p2 : lo;
            const ulong r = mod_barrett(start, p, mus[k]);   // <- no 64-bit divide
            if (r) start += (p - r);
            if ((start & 1UL) == 0) start += p;
            for (ulong m = start; m < hi; m += 2 * p)
                buf[(m - lo) >> 1] = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = lid; i < odds; i += lsz)
            if (!buf[i]) ++c;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    partial[get_global_id(0)] = c;
}
