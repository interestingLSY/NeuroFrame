#include "tensor_binary_op.h"

#include <stdexcept>
#include <cuda_runtime.h>
#include <cstdlib>

#include "src/tensor/tensor.h"
#include "src/basic/log.h"
#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

enum class BINARY_OP_TYPE {
	ADD,
	SUB,
	MUL,
	DIV,
	POW,
};

template <typename T, BINARY_OP_TYPE OP_TYPE>
__device__ __forceinline__ T perform_binary_op(const T &a, const T &b) {
	if constexpr (OP_TYPE == BINARY_OP_TYPE::ADD) {
		return a + b;
	} else if constexpr (OP_TYPE == BINARY_OP_TYPE::SUB) {
		return a - b;
	} else if constexpr (OP_TYPE == BINARY_OP_TYPE::MUL) {
		return a * b;
	} else if constexpr (OP_TYPE == BINARY_OP_TYPE::DIV) {
		return a / b;
	} else if constexpr (OP_TYPE == BINARY_OP_TYPE::POW) {
		if constexpr (std::is_same_v<T, float>) {
			return powf(a, b);
		} else if constexpr (std::is_same_v<T, double>) {
			return pow(a, b);
		} else {
			return (T)__powf((float)a, (float)b);
		}
	}
}

template <typename T, BINARY_OP_TYPE OP_TYPE>
__global__ void tensor_binary_op_kernel(
	T* __restrict__ output,
	const T* __restrict__ input1,
	const T* __restrict__ input2,
	int64_t n
) {
	#pragma unroll 4
	for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		output[i] = perform_binary_op<T, OP_TYPE>(input1[i], input2[i]);
	}
}

#define DEFINE_BINARY_OP(name, OP_TYPE) \
Tensor name(const Tensor &input1, const Tensor &input2) {\
	int64_t n = input1.numel(); \
	Tensor output(input1.shape, input1.dtype, input1.device); \
	int64_t block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE; \
	int64_t grid_size = element_wise_kernel_get_num_grids(n); \
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input1.dtype,  \
		tensor_binary_op_kernel<T, OP_TYPE><<<grid_size, block_size>>>( \
			(T*) output.data_ptr(), \
			(const T*) input1.data_ptr(), \
			(const T*) input2.data_ptr(), \
			n \
		) \
	); \
	return output; \
}

DEFINE_BINARY_OP(tensor_add, BINARY_OP_TYPE::ADD)
DEFINE_BINARY_OP(tensor_sub, BINARY_OP_TYPE::SUB)
DEFINE_BINARY_OP(tensor_mul, BINARY_OP_TYPE::MUL)
DEFINE_BINARY_OP(tensor_div, BINARY_OP_TYPE::DIV)
DEFINE_BINARY_OP(tensor_pow, BINARY_OP_TYPE::POW)


}