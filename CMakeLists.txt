# ===========================================================================
# copri -- counting primes in parallel on a single host (cross-platform build).
#
# This CMakeLists is ADDITIVE: the Unix Makefile is unchanged and remains the
# canonical build on macOS/Linux. CMake exists so the same sources build on
# Windows (MSVC + NVIDIA's OpenCL from the CUDA Toolkit), and it is the natural
# home for the future CUDA port (enable_language(CUDA)).
#
# Typical Windows use (from a "x64 Native Tools" prompt, or let run.py drive it):
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
# Binaries land in bin/<name>.exe, matching the Makefile layout run.py expects.
#
# The OpenMP versions (openmp, omp_target) are intentionally NOT built here --
# they need libomp/offload that this Windows toolchain doesn't provide. The
# Makefile still builds them where libomp is available.
# ===========================================================================
cmake_minimum_required(VERSION 3.20)
project(copri LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimized build when no config is given (single-config gens).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Emit binaries to <root>/bin (no per-config subdir), so run.py finds them the
# same way it finds the Makefile's output.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
foreach(cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${cfg}" CFG)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} "${CMAKE_SOURCE_DIR}/bin")
endforeach()

set(COMMON_DEPS
  "${CMAKE_SOURCE_DIR}/common/prime.hpp"
  "${CMAKE_SOURCE_DIR}/common/bench.hpp"
  "${CMAKE_SOURCE_DIR}/common/sieve_common.hpp")

# --- portable CPU versions: C++ standard library + threads only -------------
set(PORTABLE seq partition stripe atomic_counter atomic_dynamic sieve_cpu)

find_package(Threads REQUIRED)  # -pthread on Unix; a no-op on MSVC

foreach(v IN LISTS PORTABLE)
  add_executable(${v} "${CMAKE_SOURCE_DIR}/src/${v}.cpp" ${COMMON_DEPS})
  target_link_libraries(${v} PRIVATE Threads::Threads)
endforeach()

# --- GPU versions: opencl, sieve_gpu, sieve_gpu_barrett ---------------------
# These three are emitted under the SAME binary names from whichever backend the
# platform selects, so results from different machines carry identical version
# labels and plot/compare directly (see HANDOFF.md). Default backend: CUDA when a
# CUDA compiler is found (NVIDIA, e.g. Windows), else OpenCL (macOS framework /
# Linux ICD). Override with -DCOPRI_GPU_BACKEND=cuda|opencl|off.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(_copri_default_backend cuda)
else()
  set(_copri_default_backend opencl)
endif()
set(COPRI_GPU_BACKEND "${_copri_default_backend}" CACHE STRING
    "GPU backend for opencl/sieve_gpu/sieve_gpu_barrett: cuda, opencl, or off")

if(COPRI_GPU_BACKEND STREQUAL "cuda")
  # ---- CUDA backend (native NVIDIA) ----------------------------------------
  # nvcc compiles the kernels at build time, so there is no runtime KERNEL_PATH.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)   # RTX 2080 Ti (Turing = sm_75)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  add_executable(opencl "${CMAKE_SOURCE_DIR}/src/opencl.cu" ${COMMON_DEPS}
                 "${CMAKE_SOURCE_DIR}/src/prime_kernel.cuh"
                 "${CMAKE_SOURCE_DIR}/src/cuda_util.cuh")
  foreach(v sieve_gpu sieve_gpu_barrett)
    add_executable(${v} "${CMAKE_SOURCE_DIR}/src/${v}.cu" ${COMMON_DEPS}
                   "${CMAKE_SOURCE_DIR}/src/sieve_kernel.cuh"
                   "${CMAKE_SOURCE_DIR}/src/cuda_util.cuh")
  endforeach()
  if(WIN32)
    foreach(v opencl sieve_gpu sieve_gpu_barrett)
      target_compile_definitions(${v} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endforeach()
  endif()
  message(STATUS "GPU backend: CUDA (sm_${CMAKE_CUDA_ARCHITECTURES}) -- building opencl, sieve_gpu, sieve_gpu_barrett")

elseif(COPRI_GPU_BACKEND STREQUAL "opencl")
  # ---- OpenCL backend ------------------------------------------------------
  # On Windows the headers + OpenCL.lib come from the CUDA Toolkit; the runtime
  # ICD (OpenCL.dll) ships with the NVIDIA driver. On macOS this resolves to the
  # system OpenCL framework. If OpenCL isn't found we just skip these three.
  find_package(OpenCL QUIET)
  if(NOT OpenCL_FOUND AND WIN32)
    # Hint FindOpenCL at the CUDA Toolkit; prefer CUDA_PATH, else glob the
    # default location so the build works even from a shell predating the install.
    set(_cuda_roots "")
    if(DEFINED ENV{CUDA_PATH})
      list(APPEND _cuda_roots "$ENV{CUDA_PATH}")
    endif()
    file(GLOB _cuda_glob "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
    list(APPEND _cuda_roots ${_cuda_glob})
    foreach(_root IN LISTS _cuda_roots)
      if(EXISTS "${_root}/include/CL/cl.h" AND EXISTS "${_root}/lib/x64/OpenCL.lib")
        set(OpenCL_INCLUDE_DIR "${_root}/include" CACHE PATH "" FORCE)
        set(OpenCL_LIBRARY "${_root}/lib/x64/OpenCL.lib" CACHE FILEPATH "" FORCE)
        break()
      endif()
    endforeach()
    find_package(OpenCL QUIET)
  endif()

  if(OpenCL_FOUND)
    set(PRIME_KERNEL "${CMAKE_SOURCE_DIR}/src/prime_kernel.cl")
    set(SIEVE_KERNEL "${CMAKE_SOURCE_DIR}/src/sieve_kernel.cl")

    add_executable(opencl "${CMAKE_SOURCE_DIR}/src/opencl.cpp" ${COMMON_DEPS}
                   "${PRIME_KERNEL}")
    target_compile_definitions(opencl PRIVATE "KERNEL_PATH=\"${PRIME_KERNEL}\"")

    foreach(v sieve_gpu sieve_gpu_barrett)
      add_executable(${v} "${CMAKE_SOURCE_DIR}/src/${v}.cpp" ${COMMON_DEPS}
                     "${SIEVE_KERNEL}")
      target_compile_definitions(${v} PRIVATE "KERNEL_PATH=\"${SIEVE_KERNEL}\"")
    endforeach()

    foreach(v opencl sieve_gpu sieve_gpu_barrett)
      target_link_libraries(${v} PRIVATE OpenCL::OpenCL)
      # The host code calls clCreateCommandQueue (OpenCL 1.2); pin the API level
      # so the 3.0 headers don't error on the deprecation. Harmless on macOS too.
      target_compile_definitions(${v} PRIVATE CL_TARGET_OPENCL_VERSION=120)
      if(WIN32)
        target_compile_definitions(${v} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
      endif()
    endforeach()
    message(STATUS "GPU backend: OpenCL (${OpenCL_LIBRARY}) -- building opencl, sieve_gpu, sieve_gpu_barrett")
  else()
    message(STATUS "GPU backend: OpenCL requested but not found -- skipping GPU versions")
  endif()

else()
  message(STATUS "GPU backend: off -- skipping opencl/sieve_gpu/sieve_gpu_barrett")
endif()

message(STATUS "copri: configured. Binaries -> ${CMAKE_SOURCE_DIR}/bin")
