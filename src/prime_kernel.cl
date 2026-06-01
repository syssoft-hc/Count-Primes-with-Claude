// OpenCL kernels: count primes on the GPU, at two integer widths.
//
// is_prime_uint64 / is_prime_uint32 are byte-for-byte mirrors of the same-named
// functions in common/prime.hpp (ulong / uint instead of uint64_t / uint32_t).
// Keep them in sync. The 32-bit kernel is ~12x faster on the Apple GPU because
// that GPU has no native 64-bit integer divide -- see the README.

inline bool is_prime_uint64(ulong n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (ulong c = 3; c * c <= n; c += 2)
        if (n % c == 0) return false;
    return true;
}

inline bool is_prime_uint32(uint n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (uint c = 3; c * c <= n; c += 2)
        if (n % c == 0) return false;
    return true;
}

// Striped distribution: work-item gid tests 2+gid, 2+gid+G, ... and accumulates
// into partial[gid]; the host reduces the G partial sums. partial is ulong in
// both kernels (the GPU never does 64-bit *division* on it, only addition).

__kernel void count_primes_u64(const ulong N, const ulong G,
                               __global ulong* partial) {
    const ulong gid = get_global_id(0);
    ulong c = 0;
    for (ulong n = 2 + gid; n <= N; n += G)
        if (is_prime_uint64(n)) ++c;
    partial[gid] = c;
}

__kernel void count_primes_u32(const uint N, const uint G,
                               __global ulong* partial) {
    const uint gid = get_global_id(0);
    ulong c = 0;
    for (uint n = 2 + gid; n <= N; n += G)
        if (is_prime_uint32(n)) ++c;
    partial[gid] = c;
}
