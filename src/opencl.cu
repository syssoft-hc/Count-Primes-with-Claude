// opencl (CUDA backend) -- count primes on the GPU via trial division.
//
// This is the CUDA port of src/opencl.cpp. It keeps the binary name "opencl"
// (the project's label for "GPU trial division") so results from this NVIDIA
// machine carry identical version labels to the macOS OpenCL runs and plot/
// compare directly -- see HANDOFF.md. nvcc compiles the kernels at build time,
// so unlike the OpenCL host there is no runtime KERNEL_PATH file to load.
//
// Same striped grid + host reduction as the OpenCL version; only the GPU work
// (launch + sync + readback + reduce) is timed, matching the CPU versions.
#include "../common/bench.hpp"
#include "cuda_util.cuh"
#include "prime_kernel.cuh"
#include <vector>

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "[opencl] device: %s (CUDA, sm_%d%d)\n",
            prop.name, prop.major, prop.minor);

    const bool u32 = (a.width == Width::U32);

    // Number of GPU threads that split [2, N]; capped so each does a handful of
    // candidates for small N and the partial buffer stays small for huge N.
    unsigned long long G = 1ULL << 18;  // 262144
    if (a.N > 2 && (a.N - 1) < G) G = a.N - 1;
    if (G == 0) G = 1;
    a.threads = (unsigned)G;            // report work-items in the "threads" column

    unsigned long long* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(unsigned long long) * G));
    std::vector<unsigned long long> partial(G);

    const int block = 256;
    const unsigned int grid = (unsigned int)((G + block - 1) / block);

    // ---- timed region: launch kernel + sync + read back + reduce ----------
    int rc = run_and_report("opencl", a, [&] {
        if (u32)
            count_primes_u32<<<grid, block>>>((unsigned int)a.N, (unsigned int)G, d_partial);
        else
            count_primes_u64<<<grid, block>>>(a.N, G, d_partial);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(partial.data(), d_partial,
                              sizeof(unsigned long long) * G, cudaMemcpyDeviceToHost));
        uint64_t count = 0;
        for (unsigned long long i = 0; i < G; ++i) count += partial[i];
        return count;
    });

    cudaFree(d_partial);
    return rc;
}
