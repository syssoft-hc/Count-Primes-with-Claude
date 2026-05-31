#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <functional>
#include <thread>

// ---------------------------------------------------------------------------
// Tiny shared harness so every version behaves identically from the outside:
//
//   ./bin/<version> <N> [threads]
//
//   <N>        upper limit (inclusive); we count primes in [2, N].
//   [threads]  optional worker count; defaults to hardware_concurrency().
//
// Each binary prints exactly one machine-readable JSON line to STDOUT (consumed
// by run.py) and a human-readable summary to STDERR. Only the computation is
// timed -- argument parsing, thread creation inside the timed lambda counts as
// part of the parallel cost, which is exactly what we want to measure.
// ---------------------------------------------------------------------------

struct Args {
    uint64_t N = 1000000;  // upper limit (inclusive)
    unsigned threads = 0;  // 0 => auto-detect
};

inline Args parse_args(int argc, char** argv) {
    Args a;
    if (argc >= 2) a.N = std::strtoull(argv[1], nullptr, 10);
    if (argc >= 3) a.threads = (unsigned)std::strtoul(argv[2], nullptr, 10);
    if (a.threads == 0) {
        unsigned hc = std::thread::hardware_concurrency();
        a.threads = hc ? hc : 1;
    }
    return a;
}

// Time fn() (which returns the prime count) and report the result.
inline int run_and_report(const char* version, const Args& a,
                          const std::function<uint64_t()>& fn) {
    auto t0 = std::chrono::steady_clock::now();
    uint64_t count = fn();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // STDOUT: one JSON object, parsed by the Python runner.
    printf("{\"version\":\"%s\",\"N\":%llu,\"threads\":%u,\"count\":%llu,\"time_ms\":%.3f}\n",
           version, (unsigned long long)a.N, a.threads,
           (unsigned long long)count, ms);
    fflush(stdout);

    // STDERR: friendly summary for interactive use.
    fprintf(stderr, "[%-14s] N=%llu threads=%u -> %llu primes in %.2f ms\n",
            version, (unsigned long long)a.N, a.threads,
            (unsigned long long)count, ms);
    return 0;
}
