// partition -- static block partitioning with std::thread.
//
// Split [2, N] into P contiguous blocks, one per thread. Each thread counts the
// primes in its own block into a private slot, then the main thread sums the
// slots (a reduction with no locking).
//
// DIDACTIC POINT: this is the most obvious parallelization, but it suffers from
// LOAD IMBALANCE. is_prime(n) costs ~sqrt(n) work, so the thread that owns the
// top block does far more work than the thread that owns the bottom block and
// finishes last. Compare its speedup against `stripe` and `atomic_dynamic`.
#include "../common/prime.hpp"
#include "../common/bench.hpp"
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    return run_and_report("partition", a, [&] {
        const unsigned P = a.threads;
        const uint64_t lo = 2, hi = a.N;
        const uint64_t total = (hi >= lo) ? (hi - lo + 1) : 0;

        const Width w = a.width;
        std::vector<uint64_t> partial(P, 0);  // one private counter per thread
        std::vector<std::thread> pool;
        pool.reserve(P);

        for (unsigned t = 0; t < P; ++t) {
            // Block boundaries via t*total/P avoid rounding gaps/overlaps.
            const uint64_t begin = lo + total * t / P;
            const uint64_t end   = lo + total * (t + 1) / P;  // exclusive
            pool.emplace_back([&partial, begin, end, t, w] {
                uint64_t c = 0;
                for (uint64_t n = begin; n < end; ++n)
                    if (is_prime_w(n, w)) ++c;
                partial[t] = c;
            });
        }
        for (auto& th : pool) th.join();

        uint64_t count = 0;
        for (uint64_t v : partial) count += v;
        return count;
    });
}
