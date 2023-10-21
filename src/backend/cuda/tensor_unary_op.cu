#include "tensor_binary_op.h"

#include <stdexcept>
#include <cuda_runtime.h>
#include <cstdlib>

#include "src/tensor/tensor.h"
#include "src/basic/log.h"
#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

enum class UNARY_OP_TYPE {
	NEGATE
};

template <typename T, UNARY_OP_TYPE OP_TYPE>
__device__ __forceinline__ T perform_unary_op(const T &a) {
	if constexpr (OP_TYPE == UNARY_OP_TYPE::NEGATE) {
		return -a;
	}
}

template <typename T, UNARY_OP_TYPE OP_TYPE>
__global__ void tensor_unary_op_kernel(
	T *output,
	const T *input,
	int64_t n
) {
	#pragma unroll 4
	for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		output[i] = perform_unary_op<T, OP_TYPE>(input[i]);
	}
}

constexpr int64_t BLOCK_SIZE = 256;
const auto GRID_SIZE_CALCULATOR = [](int64_t n) {
	return std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (int64_t)16384);
};

#define DEFINE_UNARY_OP(name, OP_TYPE) \
Tensor name(const Tensor &input) {\
	int64_t n = input.numel(); \
	Tensor output(input.shape, input.dtype, input.device); \
	int64_t block_size = BLOCK_SIZE; \
	int64_t grid_size = GRID_SIZE_CALCULATOR(block_size); \
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype,  \
		tensor_unary_op_kernel<T, OP_TYPE><<<grid_size, block_size>>>( \
			(T*) output.data_ptr(), \
			(const T*) input.data_ptr(), \
			n \
		) \
	); \
	return output; \
}

DEFINE_UNARY_OP(tensor_negate, UNARY_OP_TYPE::NEGATE)


}