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
//   * even numbers are rejected immediately (2 is the only even prime),
//   * we then try odd divisors 3, 5, 7, ... up to sqrt(n).
//
// The GPU (OpenCL) version cannot include this C++ header, so it carries a
// byte-for-byte identical copy written in OpenCL C (see src/prime_kernel.cl).
// Keep the two in sync.
// ---------------------------------------------------------------------------
static inline bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (uint64_t c = 3; c * c <= n; c += 2) {
        if (n % c == 0) return false;
    }
    return true;
}
