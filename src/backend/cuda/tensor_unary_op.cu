#include "tensor_unary_op.h"

#include <stdexcept>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>

#include "src/tensor/tensor.h"
#include "src/basic/log.h"

#include "utils.h"
#include "../utils.h"

namespace NeuroFrame::Backend::CUDA {

enum class UNARY_OP_TYPE {
	NEGATE,
	INV,
	EXP,
	LOG
};

template <typename T, UNARY_OP_TYPE OP_TYPE>
__device__ __forceinline__ T perform_unary_op(const T &a) {
	if constexpr (OP_TYPE == UNARY_OP_TYPE::NEGATE) {
		return -a;
	} else if constexpr (OP_TYPE == UNARY_OP_TYPE::INV) {
		return (T)1.0 / a;
	} else if constexpr (OP_TYPE == UNARY_OP_TYPE::EXP) {
		if constexpr (std::is_same_v<T, float>) {
			return expf(a);
		} else if constexpr (std::is_same_v<T, double>) {
			return exp(a);
		} else if constexpr (std::is_same_v<T, half>) {
			return (half)exp((float)a);
		} else {
			assert(0);
		}
	} else if constexpr (OP_TYPE == UNARY_OP_TYPE::LOG) {
		if constexpr (std::is_same_v<T, float>) {
			return logf(a);
		} else if constexpr (std::is_same_v<T, double>) {
			return log(a);
		} else if constexpr (std::is_same_v<T, half>) {
			return (half)log((float)a);
		} else {
			assert(0);
		}
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

#define DEFINE_UNARY_OP(name, OP_TYPE) \
Tensor name(const Tensor &input) {\
	int64_t n = input.numel(); \
	Tensor output(input.shape, input.dtype, input.device); \
	int64_t block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE; \
	int64_t grid_size = element_wise_kernel_get_num_grids(n); \
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

DEFINE_UNARY_OP(tensor_inv, UNARY_OP_TYPE::INV)

DEFINE_UNARY_OP(tensor_exp, UNARY_OP_TYPE::EXP)

DEFINE_UNARY_OP(tensor_log, UNARY_OP_TYPE::LOG)


}