// OpenCL kernel: count primes on the GPU.
//
// is_prime_cl is a byte-for-byte mirror of is_prime() in common/prime.hpp,
// rewritten in OpenCL C (ulong instead of uint64_t). Keep the two in sync so
// the GPU runs the exact same algorithm as every CPU version.

inline bool is_prime_cl(ulong n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (ulong c = 3; c * c <= n; c += 2) {
        if (n % c == 0) return false;
    }
    return true;
}

// Striped distribution across the GPU grid: work-item `gid` tests candidates
//   2+gid, 2+gid+G, 2+gid+2G, ...   (G = global work size)
// and accumulates its local count into partial[gid]. The host reduces the
// `G` partial sums into the final answer (avoids needing 64-bit GPU atomics).
__kernel void count_primes(const ulong N,
                           const ulong G,
                           __global ulong* partial) {
    const ulong gid = get_global_id(0);
    ulong c = 0;
    for (ulong n = 2 + gid; n <= N; n += G)
        if (is_prime_cl(n)) ++c;
    partial[gid] = c;
}
