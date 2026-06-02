// sieve_gpu_barrett (CUDA backend) -- the GPU segmented sieve (see sieve_gpu.cu)
// with the per-(segment, prime) `start % p` replaced by BARRETT REDUCTION:
// two multiplies and a subtract using a per-prime mu = floor(2^64 / p)
// precomputed on the host. Same "trade division for multiplication" idea that
// uint32 used on trial division. CUDA port of src/sieve_gpu_barrett.cpp; same
// binary name for label parity with the macOS OpenCL runs (see HANDOFF.md).
#include "../common/bench.hpp"
#include "../common/sieve_common.hpp"
#include "cuda_util.cuh"
#include "sieve_kernel.cuh"
#include <vector>

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    a.width = Width::U64;   // a sieve is memory-bound; integer width is moot here
    a.width_auto = false;

    if (a.N < 3) {
        return run_and_report("sieve_gpu_barrett", a,
                              [&] { return (uint64_t)(a.N >= 2 ? 1 : 0); });
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "[sieve_gpu_barrett] device: %s (CUDA, sm_%d%d)\n",
            prop.name, prop.major, prop.minor);

    // Base primes >= 3, plus a per-prime Barrett constant mu = floor(2^64 / p).
    // For odd p (which never divides 2^64), floor((2^64-1)/p) == floor(2^64/p),
    // so the cheap host expression below is exact.
    std::vector<uint32_t> bp = base_primes(a.N);
    std::vector<unsigned int> odd_primes;
    std::vector<unsigned long long> mus;
    odd_primes.reserve(bp.size());
    mus.reserve(bp.size());
    for (uint32_t p : bp) {
        if (p < 3) continue;
        odd_primes.push_back(p);
        mus.push_back(0xFFFFFFFFFFFFFFFFULL / (unsigned long long)p);
    }
    const unsigned int nprimes = (unsigned int)odd_primes.size();

    const unsigned long long SEG_NUMS_H = 32768;
    const unsigned long long nseg = (a.N - 3) / SEG_NUMS_H + 1;

    const int WG = 256;
    unsigned long long ngrp = 2048;
    if (ngrp > nseg) ngrp = nseg ? nseg : 1;
    const unsigned long long global = ngrp * WG;
    a.threads = (unsigned)global;
    const size_t shmem = (size_t)(SEG_NUMS_H / 2);  // SEG_ODD bytes

    unsigned int* d_primes = nullptr;
    unsigned long long* d_mus = nullptr;
    unsigned long long* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_primes, sizeof(unsigned int) * (nprimes ? nprimes : 1)));
    CUDA_CHECK(cudaMalloc(&d_mus, sizeof(unsigned long long) * (nprimes ? nprimes : 1)));
    if (nprimes) {
        CUDA_CHECK(cudaMemcpy(d_primes, odd_primes.data(),
                              sizeof(unsigned int) * nprimes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mus, mus.data(),
                              sizeof(unsigned long long) * nprimes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(unsigned long long) * global));
    std::vector<unsigned long long> partial(global);

    // ---- timed region: launch kernel + sync + read back + reduce ----------
    int rc = run_and_report("sieve_gpu_barrett", a, [&] {
        sieve_count_barrett<<<(unsigned int)ngrp, WG, shmem>>>(
            a.N, nprimes, d_primes, d_mus, nseg, d_partial);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(partial.data(), d_partial,
                              sizeof(unsigned long long) * global, cudaMemcpyDeviceToHost));
        uint64_t count = 1;  // the prime 2
        for (unsigned long long i = 0; i < global; ++i) count += partial[i];
        return count;
    });

    cudaFree(d_primes);
    cudaFree(d_mus);
    cudaFree(d_partial);
    return rc;
}
