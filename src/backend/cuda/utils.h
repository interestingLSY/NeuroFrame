#pragma once 

#include <inttypes.h>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cassert>

#include "cuda_runtime.h"

#include "src/utils/cuda_utils.h"
#include "src/basic/log.h"
#include "src/backend/utils.h"

namespace NeuroFrame::Backend::CUDA {

constexpr int64_t WARP_SIZE = 32;

constexpr int64_t ELEMENT_WISE_KERNEL_BLOCK_SIZE = 256;
constexpr int64_t ELEMENT_WISE_KERNEL_MAX_GRID = 1024;

// Some stuff for indexing into an 1-D array
#define INDEX_2D(dim1, dim2, index1, index2) \
    (((int64_t)index1) * (dim2) + (index2))
#define INDEX_3D(dim1, dim2, dim3, index1, index2, index3) \
    (((int64_t)index1) * (dim2) * (dim3) + ((int64_t)index2) * (dim3) + (index3))
#define INDEX_4D(dim1, dim2, dim3, dim4, index1, index2, index3, index4) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) + ((int64_t)index2) * (dim3) * (dim4) + ((int64_t)index3) * (dim4) + (index4))
#define INDEX_5D(dim1, dim2, dim3, dim4, dim5, index1, index2, index3, index4, index5) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) * (dim5) + ((int64_t)index2) * (dim3) * (dim4) * (dim5) + ((int64_t)index3) * (dim4) * (dim5) + (index4) * (dim5) + (index5))

inline int64_t element_wise_kernel_get_num_grids(int64_t n, int64_t block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE) {
	return std::min(
		ELEMENT_WISE_KERNEL_MAX_GRID,
		(n+block_size-1) / block_size
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

template<typename T>
__device__ __forceinline__ T max(const T a, const T b) {
	if constexpr (std::is_same<T, half>::value) {
		return __hgt(a, b) ? a : b;
	} else {
		return a > b ? a : b;
	}
}

// HostToDeviceAsyncCopyBuffer: A buffer that can be used to asynchronously
// copy data from host to device.
// 
// Since cudaMemcpyAsync requires the source pointer to be a page-lock memory
// to achieve asynchronous copy, we need to allocate a page-lock memory and
// copy data from the source pointer to the page-lock memory first, and then
// copy data from the page-lock memory to the device memory.
// 
// This class implements the above idea. It allocates a page-lock memory and a
// device memory. When send() is called, it copies data from the source pointer
// to the page-lock memory, and then copies data from the page-lock memory to
// the device memory asynchronously. When wait() is called, it waits for the
// asynchronous copy to finish.
// 
// The `wait()` function is crucial for avoiding race conditions. If the
// async copy has not finished when we issued another send(), the data will
// be corrupted.
class HostToDeviceAsyncCopyBuffer {
private:
	size_t cur_size;
	cudaEvent_t copy_event;
	bool has_copy_event = false;

	void wait() {
		assert(has_copy_event);
		CUDA_CHECK(cudaEventSynchronize(copy_event));
		cudaEventDestroy(copy_event);
		has_copy_event = false;
	}

public:
	void* h_ptr;
	void* d_ptr;

	HostToDeviceAsyncCopyBuffer(const size_t init_size = 32*8) {
		assert(init_size > 0);
		cur_size = init_size;
		CUDA_CHECK(cudaMallocHost(&h_ptr, cur_size));
		CUDA_CHECK(cudaMalloc(&d_ptr, cur_size));
	}

	~HostToDeviceAsyncCopyBuffer() {
		CUDA_CHECK(cudaFreeHost(h_ptr));
		CUDA_CHECK(cudaFree(d_ptr));
	}

	void send(const void* src, const size_t size) {
		if (has_copy_event) {
			wait();
		}
		if (size > cur_size) {
			CUDA_CHECK(cudaFreeHost(h_ptr));
			CUDA_CHECK(cudaFree(d_ptr));
			while (size > cur_size) {
				cur_size *= 2;
			}
			CUDA_CHECK(cudaMallocHost(&h_ptr, cur_size));
			CUDA_CHECK(cudaMalloc(&d_ptr, cur_size));
		}
		::memcpy(h_ptr, src, size);
		CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
		cudaEventCreate(&copy_event);
		cudaEventRecord(copy_event);
		has_copy_event = true;
	}

	
};

}