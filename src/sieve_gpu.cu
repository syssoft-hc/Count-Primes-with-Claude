// sieve_gpu (CUDA backend) -- segmented Sieve of Eratosthenes on the GPU.
//
// CUDA port of src/sieve_gpu.cpp (same binary name, for label parity with the
// macOS OpenCL runs -- see HANDOFF.md). Each block sieves one segment at a time
// in fast __shared__ memory, grid-striding over all segments; the host computes
// the small base primes and sums the per-thread survivor counts (+ the prime 2).
// Like sieve_cpu, this DEPARTS from the shared-is_prime() rule on purpose.
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
        return run_and_report("sieve_gpu", a,
                              [&] { return (uint64_t)(a.N >= 2 ? 1 : 0); });
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "[sieve_gpu] device: %s (CUDA, sm_%d%d)\n",
            prop.name, prop.major, prop.minor);

    // Base primes >= 3 (the prime 2 is implicit in the odd-only representation).
    std::vector<uint32_t> bp = base_primes(a.N);
    std::vector<unsigned int> odd_primes;
    odd_primes.reserve(bp.size());
    for (uint32_t p : bp) if (p >= 3) odd_primes.push_back(p);
    const unsigned int nprimes = (unsigned int)odd_primes.size();

    // Must match the kernel's SEG_NUMS. One block sieves one segment in SEG_ODD
    // bytes of shared memory; blocks grid-stride over all segments.
    const unsigned long long SEG_NUMS_H = 32768;
    const unsigned long long nseg = (a.N - 3) / SEG_NUMS_H + 1;   // a.N >= 3 here

    const int WG = 256;                       // threads per block
    unsigned long long ngrp = 2048;           // plenty of blocks to saturate the GPU
    if (ngrp > nseg) ngrp = nseg ? nseg : 1;
    const unsigned long long global = ngrp * WG;
    a.threads = (unsigned)global;             // report work-items
    const size_t shmem = (size_t)(SEG_NUMS_H / 2);  // SEG_ODD bytes

    unsigned int* d_primes = nullptr;
    unsigned long long* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_primes, sizeof(unsigned int) * (nprimes ? nprimes : 1)));
    if (nprimes)
        CUDA_CHECK(cudaMemcpy(d_primes, odd_primes.data(),
                              sizeof(unsigned int) * nprimes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(unsigned long long) * global));
    std::vector<unsigned long long> partial(global);

    // ---- timed region: launch kernel + sync + read back + reduce ----------
    int rc = run_and_report("sieve_gpu", a, [&] {
        sieve_count<<<(unsigned int)ngrp, WG, shmem>>>(
            a.N, nprimes, d_primes, nseg, d_partial);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(partial.data(), d_partial,
                              sizeof(unsigned long long) * global, cudaMemcpyDeviceToHost));
        uint64_t count = 1;  // the prime 2
        for (unsigned long long i = 0; i < global; ++i) count += partial[i];
        return count;
    });

    cudaFree(d_primes);
    cudaFree(d_partial);
    return rc;
}
