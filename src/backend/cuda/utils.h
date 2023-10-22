#pragma once 

#include <inttypes.h>
#include <cstddef>
#include <cstdlib>
#include <cmath>

#include "cuda_runtime.h"

#include "src/basic/log.h"
#include "src/backend/utils.h"

namespace NeuroFrame::Backend::CUDA {

constexpr int ELEMENT_WISE_KERNEL_BLOCK_SIZE = 256;
constexpr int ELEMENT_WISE_KERNEL_MAX_GRID = 1024;

inline int element_wise_kernel_get_num_grids(int64_t n, int64_t block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE) {
	return std::min(
		ELEMENT_WISE_KERNEL_MAX_GRID,
		(int)((n+block_size-1) / block_size)
	);
}

// Dispatch to the correct CUDA kernel based on the dtype.
// We need a ", ## __VA_ARGS__" since that, when launching CUDA kernels, we may
// write something like kernel<<<1, 2>>>(args), which let the tokenizer think
// that the comma is a separator between arguments.
#define DISPATCH_ON_DTYPE_CUDA_BACKEND(dtype, call, ...) \
	[&]() { \
		switch (dtype) { \
			case dtype_t::FLOAT16: \
				{typedef half T; return call, ## __VA_ARGS__; } \
			case dtype_t::FLOAT32: \
				{typedef float T; return call, ## __VA_ARGS__; } \
			case dtype_t::FLOAT64: \
				{typedef double T; return call, ## __VA_ARGS__; } \
			default: \
				LOG_FATAL("Unknown dtype."); \
		} \
	}()

// get_cuda_datatype: Get the CUDA data type from the C++ type.
template<typename T>
inline cudaDataType_t get_cuda_datatype() {
    if (std::is_same<T, half>::value) {
        return CUDA_R_16F;
    } else if (std::is_same<T, float>::value) {
        return CUDA_R_32F;
    } else if (std::is_same<T, double>::value) {
        return CUDA_R_64F;
    } else {
        LOG_FATAL("Cuda data type: Unsupported type");
    }
}

template<typename T>
__host__ __device__ __forceinline__ constexpr T get_min() {
	if constexpr (std::is_same<T, half>::value) {
		return HALF_MIN;
	} else if constexpr (std::is_same<T, float>::value) {
		return FLOAT_MIN;
	} else if constexpr (std::is_same<T, double>::value) {
		return DOUBLE_MIN;
	} else {
		return 0;
	}
}

template<typename T>
__host__ __device__ __forceinline__ constexpr T get_max() {
	if constexpr (std::is_same<T, half>::value) {
		return HALF_MAX;
	} else if constexpr (std::is_same<T, float>::value) {
		return FLOAT_MAX;
	} else if constexpr (std::is_same<T, double>::value) {
		return DOUBLE_MAX;
	} else {
		return 0;
	}
}

__device__ __forceinline__ half max(const half a, const half b) {
	return __hgt(a, b) ? a : b;
}

}