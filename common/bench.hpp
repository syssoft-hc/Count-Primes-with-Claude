#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <functional>
#include <thread>
#include "prime.hpp"

// ---------------------------------------------------------------------------
// Tiny shared harness so every version behaves identically from the outside:
//
//   ./bin/<version> <N> [threads] [width]
//
//   <N>        upper limit (inclusive); we count primes in [2, N].
//   [threads]  optional worker count; defaults to hardware_concurrency().
//   [width]    optional: "u32", "u64", or "auto" (default). "auto" picks u32
//              when N <= kU32SafeMax (faster, esp. on the GPU) and u64 above it.
//
// The three optional tokens may appear in any order: a bare number is the
// thread count, "u32"/"u64"/"auto" is the width.
//
// Each binary prints exactly one machine-readable JSON line to STDOUT (consumed
// by run.py) and a human-readable summary to STDERR. Only the computation is
// timed.
// ---------------------------------------------------------------------------

enum class Width { U32, U64 };

struct Args {
    uint64_t N = 1000000;     // upper limit (inclusive)
    unsigned threads = 0;     // 0 => auto-detect
    Width width = Width::U64; // RESOLVED width actually used
    bool width_auto = true;   // true if "auto" picked the width
};

inline Args parse_args(int argc, char** argv) {
    Args a;
    enum { REQ_AUTO, REQ_U32, REQ_U64 } req = REQ_AUTO;

    if (argc >= 2) a.N = std::strtoull(argv[1], nullptr, 10);
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "u32") == 0)      req = REQ_U32;
        else if (std::strcmp(argv[i], "u64") == 0) req = REQ_U64;
        else if (std::strcmp(argv[i], "auto") == 0) req = REQ_AUTO;
        else a.threads = (unsigned)std::strtoul(argv[i], nullptr, 10);
    }
    if (a.threads == 0) {
        unsigned hc = std::thread::hardware_concurrency();
        a.threads = hc ? hc : 1;
    }

    // Resolve the effective width.
    if (req == REQ_AUTO) {
        a.width = (a.N <= kU32SafeMax) ? Width::U32 : Width::U64;
        a.width_auto = true;
    } else if (req == REQ_U32) {
        a.width_auto = false;
        if (a.N > kU32SafeMax) {
            fprintf(stderr, "[warn] u32 is unsafe for N>%llu; using u64 instead\n",
                    (unsigned long long)kU32SafeMax);
            a.width = Width::U64;
        } else {
            a.width = Width::U32;
        }
    } else {
        a.width = Width::U64;
        a.width_auto = false;
    }
    return a;
}

// Width-dispatched primality test for the CPU versions. The branch is loop-
// invariant per run and perfectly predicted; the 32-bit path still does all
// arithmetic in 32 bits, which is what matters.
static inline bool is_prime_w(uint64_t n, Width w) {
    return w == Width::U32 ? is_prime_uint32((uint32_t)n) : is_prime_uint64(n);
}

// Time fn() (which returns the prime count) and report the result.
inline int run_and_report(const char* version, const Args& a,
                          const std::function<uint64_t()>& fn) {
    auto t0 = std::chrono::steady_clock::now();
    uint64_t count = fn();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int wbits = (a.width == Width::U32) ? 32 : 64;

    // STDOUT: one JSON object, parsed by the Python runner.
    printf("{\"version\":\"%s\",\"N\":%llu,\"threads\":%u,\"width\":%d,"
           "\"count\":%llu,\"time_ms\":%.3f}\n",
           version, (unsigned long long)a.N, a.threads, wbits,
           (unsigned long long)count, ms);
    fflush(stdout);

    // STDERR: friendly summary for interactive use.
    fprintf(stderr, "[%-14s] N=%llu threads=%u width=u%d%s -> %llu primes in %.2f ms\n",
            version, (unsigned long long)a.N, a.threads, wbits,
            a.width_auto ? "(auto)" : "", (unsigned long long)count, ms);
    return 0;
}
