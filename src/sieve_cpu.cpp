// sieve_cpu -- parallel SEGMENTED SIEVE OF ERATOSTHENES (std::thread).
//
// DEPARTS from the project's shared-is_prime() rule on purpose: a sieve marks
// composites instead of dividing each candidate, so its inner loop is just
// `buf[j]=1; j+=2p` -- additions and byte writes, no division. That is why it is
// dramatically faster than every trial-division version here, and the fair
// counterpart to sieve_gpu.
//
// Structure: precompute base primes <= sqrt(N) (see sieve_common.hpp), then cut
// [3, N] into cache-sized segments over ODD numbers only. Threads pull segments
// from a shared atomic cursor (dynamic load balancing, like atomic_dynamic),
// sieve each segment in a private buffer, and count the survivors. The prime 2
// is added at the end (even numbers are never stored).
#include "../common/bench.hpp"
#include "../common/sieve_common.hpp"
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    a.width = Width::U64;   // a sieve is memory-bound; integer width is moot here
    a.width_auto = false;

    return run_and_report("sieve_cpu", a, [&] {
        const uint64_t N = a.N;
        if (N < 2) return (uint64_t)0;
        const std::vector<uint32_t> bp = base_primes(N);

        const uint64_t SEG = 1u << 18;  // numbers per segment (256K) -> 128 KB buffer
        const uint64_t nseg = (N >= 3) ? ((N - 3) / SEG + 1) : 0;
        const unsigned P = a.threads;

        std::atomic<uint64_t> cursor{0};
        std::vector<uint64_t> partial(P, 0);
        std::vector<std::thread> pool;
        pool.reserve(P);

        for (unsigned t = 0; t < P; ++t) {
            pool.emplace_back([&, t] {
                std::vector<uint8_t> buf(SEG / 2);  // one byte per odd number
                uint64_t c = 0;
                for (;;) {
                    const uint64_t s = cursor.fetch_add(1, std::memory_order_relaxed);
                    if (s >= nseg) break;
                    const uint64_t lo = 3 + s * SEG;          // odd (3 + even)
                    if (lo > N) break;
                    uint64_t hi = lo + SEG;
                    if (hi > N + 1) hi = N + 1;
                    const uint64_t odds = (hi - lo + 1) / 2;
                    std::memset(buf.data(), 0, odds);

                    for (uint32_t p : bp) {
                        if (p < 3) continue;                  // 2 handled by odd-only repr
                        const uint64_t p2 = (uint64_t)p * p;
                        if (p2 >= hi) break;                  // bp ascending -> none left
                        uint64_t start = p2 > lo ? p2 : lo;
                        const uint64_t r = start % p;
                        if (r) start += (p - r);              // first multiple of p >= start
                        if ((start & 1ULL) == 0) start += p;  // odd multiples only
                        for (uint64_t m = start; m < hi; m += 2ULL * p)
                            buf[(m - lo) >> 1] = 1;
                    }
                    for (uint64_t i = 0; i < odds; ++i)
                        if (!buf[i]) ++c;
                }
                partial[t] = c;
            });
        }
        for (auto& th : pool) th.join();

        uint64_t count = 1;  // the prime 2 (not stored in the odd-only sieve)
        for (uint64_t v : partial) count += v;
        return count;
    });
}
