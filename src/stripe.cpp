// stripe -- cyclic (striped) partitioning with std::thread.
//
// Thread t handles candidates 2+t, 2+t+P, 2+t+2P, ... i.e. every P-th number.
// Cheap small candidates and expensive large candidates are interleaved across
// all threads, so each thread sees a near-identical mix of work.
//
// DIDACTIC POINT: same number of threads as `partition`, but much better LOAD
// BALANCE because the per-candidate cost (~sqrt(n)) is spread evenly. The trade
// off is worse cache/memory locality -- here the work is compute-bound so it
// does not matter, but it would for memory-bound problems.
#include "../common/prime.hpp"
#include "../common/bench.hpp"
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    return run_and_report("stripe", a, [&] {
        const unsigned P = a.threads;
        const uint64_t lo = 2, hi = a.N;

        const Width w = a.width;
        std::vector<uint64_t> partial(P, 0);
        std::vector<std::thread> pool;
        pool.reserve(P);

        for (unsigned t = 0; t < P; ++t) {
            pool.emplace_back([&partial, hi, P, t, w] {
                uint64_t c = 0;
                for (uint64_t n = lo + t; n <= hi; n += P)
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
