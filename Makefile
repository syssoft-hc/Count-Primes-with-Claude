# ===========================================================================
# copri -- counting primes in parallel on a single host.
#
#   make            build every available version into bin/
#   make run        build, then run the Python runner with defaults
#   make clean      remove bin/
#
# Optional targets (openmp, opencl) are auto-detected: if the toolchain support
# is missing they are simply skipped, and the rest still builds.
# ===========================================================================

CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pthread
BIN      := bin
SRC      := src
COMMON   := common
DEPS     := $(COMMON)/prime.hpp $(COMMON)/bench.hpp

# --- portable versions: C++ standard library + pthreads only ----------------
PORTABLE      := seq partition stripe atomic_counter atomic_dynamic
PORTABLE_BINS := $(addprefix $(BIN)/,$(PORTABLE))

# --- OpenMP (optional): only if a libomp install (with omp.h) is found ------
LIBOMP := $(firstword $(foreach d,/opt/homebrew/opt/libomp /usr/local/opt/libomp,\
            $(if $(wildcard $(d)/include/omp.h),$(d))))
OMP_BINS :=
ifneq ($(LIBOMP),)
  OMP_BINS := $(BIN)/openmp $(BIN)/omp_target
endif

# --- OpenCL (optional): macOS framework, or libOpenCL elsewhere -------------
UNAME := $(shell uname -s)
OCL_BINS :=
ifeq ($(UNAME),Darwin)
  OCL_BINS    := $(BIN)/opencl
  OCL_LDFLAGS := -framework OpenCL
  OCL_CFLAGS  := -Wno-deprecated-declarations
endif

KERNEL := $(abspath $(SRC)/prime_kernel.cl)

ALL_BINS := $(PORTABLE_BINS) $(OMP_BINS) $(OCL_BINS)

all: $(ALL_BINS)
	@echo "built: $(notdir $(ALL_BINS))"

$(BIN):
	@mkdir -p $(BIN)

# Pattern rule for the portable versions.
$(PORTABLE_BINS): $(BIN)/%: $(SRC)/%.cpp $(DEPS) | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

# openmp (host work-sharing) and omp_target (device offload) share a recipe.
$(OMP_BINS): $(BIN)/%: $(SRC)/%.cpp $(DEPS) | $(BIN)
	$(CXX) $(CXXFLAGS) -Xpreprocessor -fopenmp \
	    -I$(LIBOMP)/include -L$(LIBOMP)/lib -lomp $< -o $@

$(BIN)/opencl: $(SRC)/opencl.cpp $(DEPS) $(SRC)/prime_kernel.cl | $(BIN)
	$(CXX) $(CXXFLAGS) $(OCL_CFLAGS) -DKERNEL_PATH='"$(KERNEL)"' \
	    $< -o $@ $(OCL_LDFLAGS)

run: all
	python3 run.py

clean:
	rm -rf $(BIN)

.PHONY: all run clean
