#include "tensor_reduction_op.h"

#include <stdexcept>
#include <cuda_runtime.h>
#include <cstdlib>

#include "src/tensor/tensor.h"
#include "src/utils/utils.h"
#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

enum class REDUTCION_OP_TYPE {
	ADD,
};

template <typename T, REDUTCION_OP_TYPE OP_TYPE>
__device__ __forceinline__ T perform_reduction_op(const T &a, const T &b) {
	if constexpr (OP_TYPE == REDUTCION_OP_TYPE::ADD) {
		return a + b;
	}
}

template <typename T, REDUTCION_OP_TYPE OP_TYPE>
__global__ void tensor_reduction_op_kernel(
	T* __restrict__ output,			// [d1, d3]
	const T* __restrict__ input,	// [d1, d2, d3]
	int64_t d1,
	int64_t d2,
	int64_t d3
) {
	// Grid size: [d1]
	// block size: [min(d3, 1024)]
	int64_t a1 = blockIdx.x;
	for (int64_t a3 = threadIdx.x; a3 < d3; a3 += blockDim.x) {
		T result = input[a1 * d2 * d3 + a3];
		#pragma unroll 4
		for (int64_t a2 = 1; a2 < d2; a2++) {
			result = perform_reduction_op<T, OP_TYPE>(result, input[INDEX_3D(d1, d2, d3, a1, a2, a3)]);
		}
		output[INDEX_2D(d1, d3, a1, a3)] = result;
	}
}

#define DEFINE_TENSOR_REDUCTION_OP(name, OP_TYPE) \
Tensor name(const Tensor &input, int axis) {\
	std::vector<int64_t> output_shape; \
	int64_t d1, d2, d3; \
	if (axis != -1) { \
		d1 = get_product_over_vector(input.shape, 0, axis); \
		d2 = input.shape[axis]; \
		d3 = get_product_over_vector(input.shape, axis + 1); \
		output_shape = input.shape; \
		output_shape.erase(output_shape.begin() + axis); \
	} else { \
		d1 = 1; \
		d2 = get_product_over_vector(input.shape); \
		d3 = 1; \
		output_shape = {}; \
	} \
	int64_t block_size = std::min(1024l, d3); \
	int64_t grid_size = d1; \
	Tensor output(output_shape, input.dtype, input.device); \
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype,  \
		tensor_reduction_op_kernel<T, OP_TYPE><<<grid_size, block_size>>>( \
			(T*) output.data_ptr(), \
			(const T*) input.data_ptr(), \
			d1, d2, d3 \
		) \
	); \
	return output; \
}

DEFINE_TENSOR_REDUCTION_OP(tensor_reduction_sum, REDUTCION_OP_TYPE::ADD)


}