// opencl -- count primes on the GPU via OpenCL.
//
// The host: picks a GPU, builds src/prime_kernel.cl at runtime, launches G
// work-items in a striped grid, then reduces the G partial sums on the CPU.
// Only the GPU work (kernel enqueue + readback) is timed, matching the CPU
// versions which time only their compute. Host-side program build is one-time
// setup, excluded from the measurement -- just like thread creation is the
// only setup the CPU versions pay.
//
// macOS ships OpenCL as a (deprecated but functional) framework; we silence the
// deprecation warnings at the build site in the Makefile.
#include "../common/bench.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifndef KERNEL_PATH
#define KERNEL_PATH "src/prime_kernel.cl"
#endif

static const char* cl_err(cl_int e); // fwd

#define CL_CHECK(expr)                                                      \
    do {                                                                    \
        cl_int _err = (expr);                                               \
        if (_err != CL_SUCCESS) {                                           \
            fprintf(stderr, "OpenCL error %d (%s) at %s:%d\n", _err,        \
                    cl_err(_err), __FILE__, __LINE__);                      \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

static std::string load_file(const char* path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open kernel file: %s\n", path);
        std::exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    // ---- one-time OpenCL setup (NOT timed) --------------------------------
    cl_platform_id platform;
    CL_CHECK(clGetPlatformIDs(1, &platform, nullptr));

    cl_device_id device;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS)
        CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr));

    char devname[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(devname), devname, nullptr);
    fprintf(stderr, "[opencl] device: %s\n", devname);

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
    cl_kernel kernel = clCreateKernel(program, "count_primes", &err);
    CL_CHECK(err);

    // Global work size: how many GPU work-items split the [2, N] range. Capped
    // so each item still does a handful of candidates even for small N, and so
    // the partial-sum buffer stays small for huge N.
    cl_ulong G = 1u << 18;  // 262144 work-items
    if (a.N > 2 && (cl_ulong)(a.N - 1) < G) G = a.N - 1;
    if (G == 0) G = 1;
    a.threads = (unsigned)G;  // report work-items in the "threads" column

    cl_mem partial_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                        sizeof(cl_ulong) * G, nullptr, &err);
    CL_CHECK(err);

    cl_ulong N = a.N;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_ulong), &N));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &G));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &partial_buf));

    std::vector<cl_ulong> partial(G);

    // ---- timed region: enqueue kernel + read back + reduce ----------------
    int rc = run_and_report("opencl", a, [&] {
        size_t global = (size_t)G;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global,
                                        nullptr, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(queue, partial_buf, CL_TRUE, 0,
                                     sizeof(cl_ulong) * G, partial.data(),
                                     0, nullptr, nullptr));
        uint64_t count = 0;
        for (cl_ulong i = 0; i < G; ++i) count += partial[i];
        return count;
    });

    clReleaseMemObject(partial_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return rc;
}

static const char* cl_err(cl_int e) {
    switch (e) {
        case CL_DEVICE_NOT_FOUND:           return "DEVICE_NOT_FOUND";
        case CL_OUT_OF_RESOURCES:           return "OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:         return "OUT_OF_HOST_MEMORY";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "MEM_ALLOC_FAILURE";
        case CL_INVALID_KERNEL_ARGS:        return "INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_GROUP_SIZE:    return "INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_VALUE:              return "INVALID_VALUE";
        case CL_BUILD_PROGRAM_FAILURE:      return "BUILD_PROGRAM_FAILURE";
        default:                            return "?";
    }
}
