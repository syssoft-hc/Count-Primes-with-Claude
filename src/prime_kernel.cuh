#pragma once
// CUDA port of src/prime_kernel.cl -- GPU trial division at two integer widths.
//
// dev_is_prime_u64 / dev_is_prime_u32 are byte-for-byte mirrors of the same
// algorithm in common/prime.hpp (and prime_kernel.cl). They are named *_dev,
// not is_prime_uint32/64, because the host header (pulled in via bench.hpp)
// already defines those names in this same translation unit -- keep all three
// copies in sync. The 32-bit path matters far less on NVIDIA than on the Apple
// GPU (x86/Turing both have real integer divide), but is kept for parity.
#include <cstdint>

__device__ inline bool dev_is_prime_u64(unsigned long long n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (unsigned long long c = 3; c * c <= n; c += 2)
        if (n % c == 0) return false;
    return true;
}

__device__ inline bool dev_is_prime_u32(unsigned int n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (unsigned int c = 3; c * c <= n; c += 2)
        if (n % c == 0) return false;
    return true;
}

// Striped distribution: thread gid tests 2+gid, 2+gid+G, ... and writes its
// count to partial[gid]; the host reduces the G partials (+0, the prime 2 is
// just another candidate here). Threads beyond G (from rounding the grid up to
// a whole number of blocks) return without touching partial.
__global__ void count_primes_u64(unsigned long long N, unsigned long long G,
                                 unsigned long long* partial) {
    const unsigned long long gid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= G) return;
    unsigned long long c = 0;
    for (unsigned long long n = 2 + gid; n <= N; n += G)
        if (dev_is_prime_u64(n)) ++c;
    partial[gid] = c;
}

__global__ void count_primes_u32(unsigned int N, unsigned int G,
                                 unsigned long long* partial) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= G) return;
    unsigned long long c = 0;
    for (unsigned int n = 2 + gid; n <= N; n += G)
        if (dev_is_prime_u32(n)) ++c;
    partial[gid] = c;
}
