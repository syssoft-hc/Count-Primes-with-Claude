// sieve_gpu_barrett -- the on-chip-blocked GPU segmented sieve (see sieve_gpu),
// but with the per-(segment, prime) `start % p` replaced by BARRETT REDUCTION.
//
// WHY: the scaling sweep (scale.py) showed `sieve_gpu` wins only in a ~1e9-1e10
// sweet spot and loses above it. The cause is that 64-bit `%`: the Apple GPU has
// no native 64-bit integer divide, and the number of base primes (hence the
// number of those divisions) grows as sqrt(N). Barrett reduction computes x % p
// with two multiplies and a subtract, given a per-prime mu = floor(2^64 / p)
// precomputed here on the host (where 64-bit division is cheap). It is the same
// "trade division for multiplication" idea that rescued trial division via
// uint32 -- applied to reclaim the large-N regime for the GPU sieve.
//
// Everything else (grid-strided work-groups, __local segment blocking, odd-only
// representation, host reduction + the prime 2) is identical to sieve_gpu.
#include "../common/bench.hpp"
#include "../common/sieve_common.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifndef KERNEL_PATH
#define KERNEL_PATH "src/sieve_kernel.cl"
#endif

static const char* cl_err(cl_int e);

#define CL_CHECK(expr)                                                      \
    do {                                                                    \
        cl_int _err = (expr);                                              \
        if (_err != CL_SUCCESS) {                                          \
            fprintf(stderr, "OpenCL error %d (%s) at %s:%d\n", _err,        \
                    cl_err(_err), __FILE__, __LINE__);                     \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

static std::string load_file(const char* path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open kernel file: %s\n", path); std::exit(1); }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    a.width = Width::U64;   // a sieve is memory-bound; integer width is moot here
    a.width_auto = false;

    if (a.N < 3) {
        return run_and_report("sieve_gpu_barrett", a,
                              [&] { return (uint64_t)(a.N >= 2 ? 1 : 0); });
    }

    // ---- one-time OpenCL setup (NOT timed) --------------------------------
    cl_platform_id platform;
    CL_CHECK(clGetPlatformIDs(1, &platform, nullptr));
    cl_device_id device;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS)
        CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr));
    char devname[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(devname), devname, nullptr);
    fprintf(stderr, "[sieve_gpu_barrett] device: %s\n", devname);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    CL_CHECK(err);

    std::string src = load_file(KERNEL_PATH);
    const char* src_ptr = src.c_str();
    size_t src_len = src.size();
    cl_program program = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    CL_CHECK(err);
    if (clBuildProgram(program, 1, &device, "", nullptr, nullptr) != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::vector<char> log(logsz + 1, 0);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        fprintf(stderr, "kernel build failed:\n%s\n", log.data());
        std::exit(1);
    }
    cl_kernel kernel = clCreateKernel(program, "sieve_count_barrett", &err);
    CL_CHECK(err);

    // Base primes >= 3, plus a per-prime Barrett constant mu = floor(2^64 / p).
    // For odd p (which never divides 2^64), floor((2^64-1)/p) == floor(2^64/p),
    // so the cheap host expression below is exact.
    std::vector<uint32_t> bp = base_primes(a.N);
    std::vector<cl_uint> odd_primes;
    std::vector<cl_ulong> mus;
    odd_primes.reserve(bp.size());
    mus.reserve(bp.size());
    for (uint32_t p : bp) {
        if (p < 3) continue;
        odd_primes.push_back(p);
        mus.push_back((cl_ulong)0xFFFFFFFFFFFFFFFFULL / (cl_ulong)p);
    }
    const cl_uint nprimes = (cl_uint)odd_primes.size();

    const cl_ulong SEG_NUMS = 32768;
    const cl_ulong SEG_ODD = SEG_NUMS / 2;
    const cl_ulong nseg = (a.N - 3) / SEG_NUMS + 1;

    const size_t WG = 256;
    cl_ulong ngrp = 2048;
    if (ngrp > nseg) ngrp = nseg ? nseg : 1;
    const size_t global = (size_t)(ngrp * WG);
    a.threads = (unsigned)global;

    cl_mem primes_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(cl_uint) * (nprimes ? nprimes : 1),
                                       odd_primes.empty() ? nullptr : odd_primes.data(), &err);
    CL_CHECK(err);
    cl_mem mus_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_ulong) * (nprimes ? nprimes : 1),
                                    mus.empty() ? nullptr : mus.data(), &err);
    CL_CHECK(err);
    cl_mem partial_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_ulong) * global, nullptr, &err);
    CL_CHECK(err);

    cl_ulong N = a.N;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_ulong), &N));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_uint), &nprimes));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &primes_buf));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &mus_buf));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &nseg));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &partial_buf));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uchar) * SEG_ODD, nullptr));  // __local

    std::vector<cl_ulong> partial(global);

    // ---- timed region: enqueue kernel + read back + reduce ----------------
    int rc = run_and_report("sieve_gpu_barrett", a, [&] {
        size_t gws = global, lws = WG;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws, &lws,
                                        0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(queue, partial_buf, CL_TRUE, 0,
                                     sizeof(cl_ulong) * global, partial.data(),
                                     0, nullptr, nullptr));
        uint64_t count = 1;  // the prime 2
        for (size_t i = 0; i < global; ++i) count += partial[i];
        return count;
    });

    clReleaseMemObject(primes_buf);
    clReleaseMemObject(mus_buf);
    clReleaseMemObject(partial_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return rc;
}

static const char* cl_err(cl_int e) {
    switch (e) {
        case CL_DEVICE_NOT_FOUND:              return "DEVICE_NOT_FOUND";
        case CL_OUT_OF_RESOURCES:              return "OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:            return "OUT_OF_HOST_MEMORY";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "MEM_ALLOC_FAILURE";
        case CL_INVALID_KERNEL_ARGS:           return "INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_GROUP_SIZE:       return "INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_BUFFER_SIZE:           return "INVALID_BUFFER_SIZE";
        case CL_INVALID_VALUE:                 return "INVALID_VALUE";
        case CL_BUILD_PROGRAM_FAILURE:         return "BUILD_PROGRAM_FAILURE";
        default:                               return "?";
    }
}
