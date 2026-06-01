#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// Shared primality test used by EVERY CPU version in this project.
//
// The whole point of this repository is to study how to exploit parallelism on
// a single host, NOT how to test primality quickly. So we deliberately use the
// simplest correct method: trial division. A candidate n is prime iff no odd
// number c with c*c <= n divides it.
//
// We expose the SAME algorithm at two fixed widths -- is_prime_uint32() and
// is_prime_uint64() -- because the integer width matters enormously on the GPU:
// the Apple GPU has no native 64-bit integer divide, so the 64-bit `%` is
// emulated and ~12x slower than the 32-bit one (the CPU barely cares; its
// 64-bit divide is nearly as fast). See the README "uint32 vs uint64" section.
//
// The single source of truth is the template is_prime_impl<T>; the two named
// entry points are thin wrappers so there is still exactly one implementation.
// The GPU (OpenCL) version cannot include this header, so src/prime_kernel.cl
// carries a byte-for-byte mirror of both. Keep them in sync.
// ---------------------------------------------------------------------------
template <typename T>
static inline bool is_prime_impl(T n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (T c = 3; c * c <= n; c += 2) {
        if (n % c == 0) return false;
    }
    return true;
}

static inline bool is_prime_uint32(uint32_t n) { return is_prime_impl<uint32_t>(n); }
static inline bool is_prime_uint64(uint64_t n) { return is_prime_impl<uint64_t>(n); }

// Largest N for which the uint32 path is guaranteed overflow-safe. The loop
// evaluates c*c for c up to ~sqrt(N)+2, and that product must stay below 2^32.
// 4e9 leaves comfortable margin: sqrt(4e9)+2, squared, is ~4.0002e9 < 2^32.
// Above this, callers must use the uint64 path (and above 2^32 a uint32 simply
// cannot even represent the candidate).
static constexpr uint64_t kU32SafeMax = 4000000000ULL;
