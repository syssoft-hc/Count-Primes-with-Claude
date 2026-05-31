// atomic_dynamic -- dynamic work distribution via an atomic chunk dispatcher.
//
// Instead of statically assigning work up front, threads cooperatively pull the
// next chunk of candidates from a shared std::atomic<uint64_t> cursor:
//
//     base = next.fetch_add(CHUNK);   // claim [base, base+CHUNK)
//
// Whenever a thread finishes a chunk it grabs another, so a thread that happens
// to draw expensive (large) candidates simply claims fewer chunks. This is the
// "atomic done right": one atomic op per CHUNK (not per prime), giving both good
// load balance AND negligible contention. The prime count is kept locally per
// thread and reduced once at the end.
//
// DIDACTIC POINT: this is essentially a hand-rolled OpenMP `schedule(dynamic)`.
// CHUNK trades scheduling overhead (small chunks = more atomic ops) against load
// balance (large chunks = more tail imbalance).
#include "../common/prime.hpp"
#include "../common/bench.hpp"
#include <atomic>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    return run_and_report("atomic_dynamic", a, [&] {
        const unsigned P = a.threads;
        const uint64_t lo = 2, hi = a.N;
        const uint64_t CHUNK = 4096;  // candidates claimed per atomic op

        std::atomic<uint64_t> cursor{lo};
        std::vector<uint64_t> partial(P, 0);
        std::vector<std::thread> pool;
        pool.reserve(P);

        for (unsigned t = 0; t < P; ++t) {
            pool.emplace_back([&cursor, &partial, hi, t] {
                uint64_t c = 0;
                for (;;) {
                    const uint64_t base = cursor.fetch_add(CHUNK, std::memory_order_relaxed);
                    if (base > hi) break;
                    const uint64_t end = (base + CHUNK <= hi + 1) ? base + CHUNK : hi + 1;
                    for (uint64_t n = base; n < end; ++n)
                        if (is_prime(n)) ++c;
                }
                partial[t] = c;
            });
        }
        for (auto& th : pool) th.join();

        uint64_t count = 0;
        for (uint64_t v : partial) count += v;
        return count;
    });
}
