// seq -- sequential baseline.
//
// One thread walks every candidate in [2, N] and counts the primes. This is the
// reference both for correctness (every parallel version must produce the same
// count) and for speed (every parallel version is measured against this time).
#include "../common/prime.hpp"
#include "../common/bench.hpp"

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    a.threads = 1;  // by definition

    return run_and_report("seq", a, [&] {
        uint64_t count = 0;
        for (uint64_t n = 2; n <= a.N; ++n)
            if (is_prime(n)) ++count;
        return count;
    });
}
