#include "tensor_eq.h"

#include <cuda_fp16.h>

#include "utils.h"
#include "src/backend/utils.h"	// For HALF_ABS_THRES, HALF_REL_THRES, etc.

namespace NeuroFrame::Backend::CUDA {

template<typename T>
__device__ T __abs(const T &a) {
	if constexpr (std::is_same_v<T, half>) {
		return __habs(a);
	} else {
		return abs(a);
	}
}

template<typename T>
__device__ T __max(const T &a, const T &b) {
	return a > b ? a : b;
}

template<typename T>
__device__ bool is_elem_eq(const T &a, const T &b) {
	T abs_err = __abs(a - b);
	T rel_err = abs_err / (__max(__abs(a), __abs(b)) + (T)(1e-5));
	if constexpr (std::is_same_v<T, half>) {
		return abs_err <= HALF_ABS_THRES && rel_err <= HALF_REL_THRES;
	} else if constexpr (std::is_same_v<T, float>) {
		return abs_err <= FLOAT_ABS_THRES && rel_err <= FLOAT_REL_THRES;
	} else if constexpr (std::is_same_v<T, double>) {
		return abs_err <= DOUBLE_ABS_THRES && rel_err <= DOUBLE_REL_THRES;
	} else {
		return a == b;
	}
} 

// This variable is used to store the result of the comparison
// The whole procedure of `tensor_eq` looks like this:
// - `tensor_eq` picks up a number, which has never been used before
// - `tensor_eq` calls `tensor_eq_kernel`. If the kernel finds a mismatch, it writes
//   the number we picked up to `result`
// - `tensor_eq` checks the value of `result`. If it is the number we picked up,
//   then there is a mismatch, and `tensor_eq` returns false. Otherwise, it returns true.
// This eliminates unnecessary atomic operations or memory copies.
static __device__ int64_t result = 0;

template<typename T>
__global__ void tensor_eq_kernel(
	const T* __restrict__ arr1,
	const T* __restrict__ arr2,
	const int64_t n,
	const int64_t tag	// The number we picked up
) {
	#pragma unroll 4
	for (int64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
		if (!is_elem_eq(arr1[i], arr2[i]))
			result = tag;
}

bool tensor_eq(const Tensor &input1, const Tensor &input2) {
	// The two tensors are guaranteed to have the same shape, data type, and device
	static int64_t tag = 1;
	tag += 1;
	int block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE;
	int grid_size = element_wise_kernel_get_num_grids(input1.numel(), block_size);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input1.dtype, tensor_eq_kernel<<<grid_size, block_size>>>(
		(const T*)input1.data_ptr(),
		(const T*)input2.data_ptr(),
		input1.numel(),
		tag
	));
	int64_t result_host;
	cudaMemcpyFromSymbol(&result_host, result, sizeof(int64_t), 0, cudaMemcpyDeviceToHost);
	return result_host != tag;	// Return true when no thread puts `result` to `tag`
}

}