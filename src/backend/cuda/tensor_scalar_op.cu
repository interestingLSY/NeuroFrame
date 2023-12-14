#include "tensor_scalar_op.h"

#include <stdexcept>
#include <cuda_runtime.h>
#include <cstdlib>

#include "src/tensor/tensor.h"
#include "src/basic/log.h"
#include "src/basic/scalar.h"
#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

enum class TENSOR_SCALAR_OP_TYPE {
	ADDS
};

template <typename T, TENSOR_SCALAR_OP_TYPE OP_TYPE>
__device__ __forceinline__ T perform_tensor_scalar_op(const T &tensor_elem, const T &scalar_val) {
	if constexpr (OP_TYPE == TENSOR_SCALAR_OP_TYPE::ADDS) {
		return tensor_elem + scalar_val;
	}
}

template <typename T, TENSOR_SCALAR_OP_TYPE OP_TYPE>
__global__ void tensor_scalar_op_kernel(
	T *output,
	const T *input,
	const T &scalar_val,
	int64_t n
) {
	#pragma unroll 4
	for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		output[i] = perform_tensor_scalar_op<T, OP_TYPE>(input[i], scalar_val);
	}
}

constexpr int64_t BLOCK_SIZE = 256;
const auto GRID_SIZE_CALCULATOR = [](int64_t n) {
	return std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (int64_t)16384);
};

#define DEFINE_TENSOR_SCALAR_OP(NAME, OP_TYPE) \
Tensor NAME(const Tensor &input, const Scalar &scalar) {\
	int64_t n = input.numel(); \
	Tensor output(input.shape, input.dtype, input.device); \
	int64_t block_size = BLOCK_SIZE; \
	int64_t grid_size = GRID_SIZE_CALCULATOR(block_size); \
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype,  \
		tensor_scalar_op_kernel<T, OP_TYPE><<<grid_size, block_size>>>( \
			(T*) output.data_ptr(), \
			(const T*) input.data_ptr(), \
			scalar.to_c_dtype<T>(), \
			n \
		) \
	); \
	return output; \
}

DEFINE_TENSOR_SCALAR_OP(tensor_adds, TENSOR_SCALAR_OP_TYPE::ADDS)


}