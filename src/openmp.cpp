// openmp -- the same computation expressed with OpenMP pragmas.
//
// This is the high-level counterpart to the hand-rolled std::thread versions:
// the runtime creates the team, schedule(dynamic) gives the load balancing that
// `atomic_dynamic` builds by hand, and reduction(+:count) gives the private-
// accumulator-then-reduce pattern that `stripe` builds by hand. A few lines of
// pragma replace all of that bookkeeping.
//
// OPTIONAL TARGET: Apple clang needs a separate libomp (e.g. `brew install
// libomp`). The Makefile only builds this if libomp is found, and run.py simply
// skips it when the binary is absent.
#include "../common/prime.hpp"
#include "../common/bench.hpp"
#include <omp.h>

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    omp_set_num_threads((int)a.threads);

    const Width w = a.width;
    return run_and_report("openmp", a, [&] {
        uint64_t count = 0;
        const long long N = (long long)a.N;  // OpenMP wants a signed loop var
        #pragma omp parallel for schedule(dynamic, 4096) reduction(+ : count)
        for (long long n = 2; n <= N; ++n)
            if (is_prime_w((uint64_t)n, w)) ++count;
        return count;
    });
}
