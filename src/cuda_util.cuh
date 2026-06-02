#pragma once
// Shared CUDA helpers for the .cu ports (opencl.cu, sieve_gpu.cu,
// sieve_gpu_barrett.cu). Header-only; included by exactly one .cu per binary.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Abort with a readable message on any CUDA API failure -- the analogue of the
// OpenCL versions' CL_CHECK macro.
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", (int)_e,          \
                    cudaGetErrorString(_e), __FILE__, __LINE__);               \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
