// omp_target -- OpenMP GPU *offload* version (#pragma omp target).
//
// This is the offload counterpart to `openmp`. Instead of a host work-sharing
// loop, the candidate range is shipped to an OpenMP *device* with
//     #pragma omp target teams distribute parallel for
// which (on capable hardware) runs on the GPU: `teams` map to GPU thread blocks,
// `distribute parallel for` spreads iterations across them, and reduction(+)
// combines the per-thread counts.
//
// HONEST CAVEAT FOR THIS PROJECT'S MAIN HOST (Apple Silicon):
//   There is NO OpenMP offload backend for the Apple GPU (OpenMP offloads to
//   NVPTX/AMDGPU/SPIR-V; the Apple GPU is programmed via Metal/OpenCL only).
//   So here omp_get_num_devices() == 0 and the target region transparently
//   FALLS BACK TO THE HOST CPU. The exact same source offloads to the GPU on a
//   machine with an NVIDIA/AMD GPU and an offload-enabled LLVM, e.g.:
//       clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda ...
//   The program detects and PRINTS which actually happened, so the benchmark is
//   never silently mislabeled. For the real GPU path on this Mac, see `opencl`.
//
// The shared is_prime() is reused unchanged: wrapping its include in a
// `declare target` region makes the compiler also emit a device-side copy, so
// there is still exactly one primality implementation.
#include "../common/bench.hpp"
#include <omp.h>

#pragma omp declare target
#include "../common/prime.hpp"
#pragma omp end declare target

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    // Detect, outside the timed region, where a target region actually executes.
    const int ndev = omp_get_num_devices();
    int on_host = 1;
    #pragma omp target map(tofrom : on_host)
    {
        on_host = omp_is_initial_device();
    }
    const bool offloaded = !on_host;
    fprintf(stderr, "[omp_target] offload devices=%d -> target runs on %s\n",
            ndev, offloaded ? "DEVICE (GPU)" : "HOST (CPU fallback)");
    // If we fell back to the host, honor the requested thread count so the run
    // is still a meaningful CPU measurement; on a real device this is ignored.
    if (!offloaded) omp_set_num_threads((int)a.threads);

    const int w32 = (a.width == Width::U32) ? 1 : 0;
    return run_and_report("omp_target", a, [&] {
        uint64_t count = 0;
        const long long N = (long long)a.N;
        // is_prime_uint32/64 are device-available (prime.hpp is included inside
        // a declare target region below); pick per-candidate via the mapped flag.
        #pragma omp target teams distribute parallel for \
            reduction(+ : count) map(tofrom : count) firstprivate(w32)
        for (long long n = 2; n <= N; ++n) {
            bool p = w32 ? is_prime_uint32((uint32_t)n) : is_prime_uint64((uint64_t)n);
            if (p) ++count;
        }
        return count;
    });
}
