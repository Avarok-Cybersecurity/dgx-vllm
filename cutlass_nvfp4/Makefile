# Makefile for Custom CUTLASS NVFP4 Kernel
# Target: NVIDIA GB10 (SM_121, Blackwell)

# CUDA paths
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

# Compiler flags
NVCC_FLAGS := -std=c++17 \
              -arch=sm_121 \
              -gencode=arch=compute_121,code=sm_121 \
              -O3 \
              -use_fast_math \
              -lineinfo \
              -Xptxas=-v

# Include paths
INCLUDES := -I$(CUDA_PATH)/include \
            -I. \
            -I/workspace/cutlass/include

# Targets
all: test_nvfp4_types test_nvfp4_gemm test_nvfp4_gemm_hardware test_ptx_v1 test_ptx_v2 benchmark_fp4 test_optimized_kernel

test_nvfp4_types: test_nvfp4_types.cu nvfp4_types.cuh
	@echo "Compiling NVFP4 data type tests..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_nvfp4_types.cu -o test_nvfp4_types
	@echo "✓ Compilation successful!"

test_nvfp4_gemm: test_nvfp4_gemm.cu nvfp4_gemm_kernel.cuh nvfp4_types.cuh
	@echo "Compiling NVFP4 GEMM kernel tests..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_nvfp4_gemm.cu -o test_nvfp4_gemm
	@echo "✓ Compilation successful!"

test_nvfp4_gemm_hardware: test_nvfp4_gemm_hardware.cu nvfp4_gemm_kernel_hardware.cuh nvfp4_gemm_kernel.cuh nvfp4_types.cuh
	@echo "Compiling NVFP4 HARDWARE tensor core kernel tests..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_nvfp4_gemm_hardware.cu -o test_nvfp4_gemm_hardware
	@echo "✓ Compilation successful!"

test_ptx_v1: test_ptx_v1.cu nvfp4_tcgen05_ptx.cuh nvfp4_gemm_kernel.cuh nvfp4_types.cuh
	@echo "Compiling PTX V1 infrastructure test..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_ptx_v1.cu -o test_ptx_v1
	@echo "✓ Compilation successful!"

test_ptx_v2: test_ptx_v2.cu nvfp4_tcgen05_ptx_v2.cuh nvfp4_gemm_kernel.cuh nvfp4_types.cuh
	@echo "Compiling PTX V2 REAL HARDWARE test..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_ptx_v2.cu -o test_ptx_v2
	@echo "✓ Compilation successful!"

run_tests: test_nvfp4_types
	@echo ""
	@echo "Running NVFP4 data type unit tests..."
	@echo ""
	./test_nvfp4_types

run_gemm_tests: test_nvfp4_gemm
	@echo ""
	@echo "Running NVFP4 GEMM kernel tests..."
	@echo ""
	./test_nvfp4_gemm

run_hardware_tests: test_nvfp4_gemm_hardware
	@echo ""
	@echo "Running NVFP4 HARDWARE tensor core tests..."
	@echo ""
	./test_nvfp4_gemm_hardware

run_ptx_v1: test_ptx_v1
	@echo ""
	@echo "Running PTX V1 infrastructure tests..."
	@echo ""
	./test_ptx_v1

run_ptx_v2: test_ptx_v2
	@echo ""
	@echo "🚀 Running PTX V2 REAL HARDWARE tests..."
	@echo ""
	./test_ptx_v2

run_all_tests: run_tests run_gemm_tests run_hardware_tests run_ptx_v1 run_ptx_v2
	@echo ""
	@echo "✓ All test suites completed"

benchmark_fp4: benchmark_fp4.cu nvfp4_types.cuh nvfp4_gemm_simple_hw.cuh cuda_fp4.h
	@echo "Compiling FP4 benchmark suite..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) benchmark_fp4.cu -o benchmark_fp4 -lcublas
	@echo "✓ Compilation successful!"

run_benchmark: benchmark_fp4
	@echo ""
	@echo "🚀 Running FP4 Performance Benchmark..."
	@echo ""
	./benchmark_fp4

test_optimized_kernel: test_optimized_kernel.cu nvfp4_gemm_kernel_optimized.cuh nvfp4_gemm_kernel.cuh nvfp4_tcgen05_ptx_v2.cuh nvfp4_types.cuh
	@echo "Compiling OPTIMIZED NVFP4 kernel (ALL 4 OPTIMIZATIONS)..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) test_optimized_kernel.cu -o test_optimized_kernel
	@echo "✅ Optimized kernel compiled - READY TO BEAT AWQ!"

run_optimized_test: test_optimized_kernel
	@echo ""
	@echo "🔥 TEAM NVIDIA vs TEAM AWQ - OPTIMIZED KERNEL TEST 🔥"
	@echo ""
	./test_optimized_kernel

clean:
	rm -f test_nvfp4_types test_nvfp4_gemm test_nvfp4_gemm_hardware test_ptx_v1 test_ptx_v2 benchmark_fp4 test_optimized_kernel *.o

.PHONY: all run_tests run_gemm_tests run_hardware_tests run_ptx_v1 run_ptx_v2 run_all_tests clean
