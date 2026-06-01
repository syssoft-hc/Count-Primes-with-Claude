// atomic_counter -- shared std::atomic result counter (the naive atomic).
//
// Work is striped exactly like `stripe`, but instead of each thread keeping a
// private counter and reducing at the end, EVERY prime found does a
// fetch_add on ONE global std::atomic<uint64_t>.
//
// DIDACTIC POINT: this is the textbook "just make it atomic" reflex, and it is
// usually the WRONG granularity. Each increment forces cache-line ownership to
// bounce between cores (contention on a single hot line). Compare its time to
// `stripe`, which does the identical work but accumulates locally and reduces
// once -- the only difference is where the atomic is, and it costs real time.
//
// (We also show the right fix in comments: accumulate into a local, add once.)
#include "../common/prime.hpp"
#include "../common/bench.hpp"
#include <atomic>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    return run_and_report("atomic_counter", a, [&] {
        const unsigned P = a.threads;
        const uint64_t lo = 2, hi = a.N;

        const Width w = a.width;
        std::atomic<uint64_t> count{0};
        std::vector<std::thread> pool;
        pool.reserve(P);

        for (unsigned t = 0; t < P; ++t) {
            pool.emplace_back([&count, hi, P, t, w] {
                for (uint64_t n = lo + t; n <= hi; n += P)
                    if (is_prime_w(n, w))
                        // Hot, contended increment on a shared cache line.
                        count.fetch_add(1, std::memory_order_relaxed);
                // The cheap alternative would be:
                //     uint64_t local = 0; ... ++local ...
                //     count.fetch_add(local, std::memory_order_relaxed);
                // i.e. one atomic op per THREAD instead of one per PRIME.
            });
        }
        for (auto& th : pool) th.join();
        return count.load();
    });
}
