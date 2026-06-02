#pragma once
// CUDA port of src/sieve_kernel.cl -- segmented Sieve of Eratosthenes blocked in
// on-chip __shared__ memory (CUDA's analogue of OpenCL __local). One BLOCK
// sieves one segment at a time, grid-striding over all segments; odd numbers
// only (slot i in a segment is number segLo + 2*i); the prime 2 is added on the
// host. Inner loop is `buf[..]=1; m += 2p` -- division-free. Benign byte-store
// races (always writing 1) mean no atomics are needed. See sieve_kernel.cl for
// the full rationale.
//
// Translation from OpenCL (per HANDOFF.md):
//   get_local_id(0)  -> threadIdx.x      get_group_id(0)  -> blockIdx.x
//   get_local_size(0)-> blockDim.x       get_num_groups(0)-> gridDim.x
//   __local uchar* buf  -> extern __shared__ unsigned char buf[] (+launch shmem)
//   barrier(CLK_LOCAL_MEM_FENCE) -> __syncthreads()
//   mul_hi(a,b)       -> __umul64hi(a,b)

#define SEG_NUMS 32768u            // numbers per segment (even)
#define SEG_ODD  (SEG_NUMS / 2u)   // odd slots per segment -> shared bytes

// buf is the dynamically-sized shared array, passed as the 3rd launch argument
// (SEG_ODD bytes). All threads in a block share it.

__global__ void sieve_count(unsigned long long N,
                            unsigned int nprimes,
                            const unsigned int* __restrict__ primes,  // odd base primes >= 3
                            unsigned long long nseg,
                            unsigned long long* __restrict__ partial) {
    extern __shared__ unsigned char buf[];
    const unsigned int lid = threadIdx.x;
    const unsigned int lsz = blockDim.x;
    const unsigned long long ngrp = gridDim.x;
    unsigned long long c = 0;

    for (unsigned long long s = blockIdx.x; s < nseg; s += ngrp) {
        const unsigned long long lo = 3 + s * (unsigned long long)SEG_NUMS;
        if (lo > N) break;                       // uniform across the block
        unsigned long long hi = lo + SEG_NUMS;
        if (hi > N + 1) hi = N + 1;
        const unsigned long long odds = (hi - lo + 1) / 2;

        for (unsigned int i = lid; i < odds; i += lsz) buf[i] = 0;
        __syncthreads();

        for (unsigned int k = lid; k < nprimes; k += lsz) {
            const unsigned long long p = primes[k];
            const unsigned long long p2 = p * p;
            if (p2 >= hi) continue;              // strided k -> continue, not break
            unsigned long long start = p2 > lo ? p2 : lo;
            const unsigned long long r = start % p;
            if (r) start += (p - r);             // first multiple of p >= start
            if ((start & 1ULL) == 0) start += p; // odd multiples only
            for (unsigned long long m = start; m < hi; m += 2 * p)
                buf[(m - lo) >> 1] = 1;
        }
        __syncthreads();

        for (unsigned int i = lid; i < odds; i += lsz)
            if (!buf[i]) ++c;                    // segment starts at 3, no "1" to exclude
        __syncthreads();                         // reuse buf next iteration
    }

    partial[(unsigned long long)blockIdx.x * blockDim.x + threadIdx.x] = c;
}

// ---------------------------------------------------------------------------
// sieve_count_barrett: identical, but the per-(segment, prime) `start % p` is
// replaced by Barrett reduction -- two multiplies and a subtract using a
// per-prime mu = floor(2^64 / p) precomputed on the host. mul_hi -> __umul64hi.
__device__ inline unsigned long long mod_barrett(unsigned long long x,
                                                 unsigned long long p,
                                                 unsigned long long mu) {
    unsigned long long q = __umul64hi(x, mu);    // high 64 bits of x*mu ~ floor(x/p)
    unsigned long long r = x - q * p;            // x mod p, off by up to ~2p
    while (r >= p) r -= p;                        // normalize (<= ~2 iterations)
    return r;
}

__global__ void sieve_count_barrett(unsigned long long N,
                                    unsigned int nprimes,
                                    const unsigned int*       __restrict__ primes,
                                    const unsigned long long* __restrict__ mus,
                                    unsigned long long nseg,
                                    unsigned long long* __restrict__ partial) {
    extern __shared__ unsigned char buf[];
    const unsigned int lid = threadIdx.x;
    const unsigned int lsz = blockDim.x;
    const unsigned long long ngrp = gridDim.x;
    unsigned long long c = 0;

    for (unsigned long long s = blockIdx.x; s < nseg; s += ngrp) {
        const unsigned long long lo = 3 + s * (unsigned long long)SEG_NUMS;
        if (lo > N) break;
        unsigned long long hi = lo + SEG_NUMS;
        if (hi > N + 1) hi = N + 1;
        const unsigned long long odds = (hi - lo + 1) / 2;

        for (unsigned int i = lid; i < odds; i += lsz) buf[i] = 0;
        __syncthreads();

        for (unsigned int k = lid; k < nprimes; k += lsz) {
            const unsigned long long p = primes[k];
            const unsigned long long p2 = p * p;
            if (p2 >= hi) continue;
            unsigned long long start = p2 > lo ? p2 : lo;
            const unsigned long long r = mod_barrett(start, p, mus[k]);  // no 64-bit divide
            if (r) start += (p - r);
            if ((start & 1ULL) == 0) start += p;
            for (unsigned long long m = start; m < hi; m += 2 * p)
                buf[(m - lo) >> 1] = 1;
        }
        __syncthreads();

        for (unsigned int i = lid; i < odds; i += lsz)
            if (!buf[i]) ++c;
        __syncthreads();
    }

    partial[(unsigned long long)blockIdx.x * blockDim.x + threadIdx.x] = c;
}
