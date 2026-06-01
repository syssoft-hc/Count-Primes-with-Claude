#pragma once
#include <cstdint>
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// "Base primes" up to sqrt(N), shared by the CPU and GPU segmented sieves.
//
// A segmented sieve marks the composites in a range by striking out multiples
// of every prime <= sqrt(N). Those small primes are produced once here with a
// plain sieve -- the list is tiny (sqrt(1e9) ~ 31623, ~3400 primes), so this
// step is negligible next to sieving the full [2, N].
//
// NOTE: the sieve versions intentionally DEPART from the project's "every
// version shares is_prime()" rule -- a sieve marks composites instead of testing
// each candidate, which is why its inner loop is division-free and so much
// faster (and why the GPU finally wins). See the README.
// ---------------------------------------------------------------------------
inline std::vector<uint32_t> base_primes(uint64_t N) {
    const uint32_t r = (uint32_t)std::sqrt((double)N) + 1;
    std::vector<uint8_t> comp(r + 1, 0);
    std::vector<uint32_t> primes;
    for (uint32_t p = 2; p <= r; ++p) {
        if (!comp[p]) {
            primes.push_back(p);
            for (uint64_t m = (uint64_t)p * p; m <= r; m += p) comp[m] = 1;
        }
    }
    return primes;
}
