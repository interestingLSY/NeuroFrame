#include "tensor_reduction_op.h"

#include "utils.h"
#include "src/utils/utils.h"

namespace NeuroFrame::Backend::CPU {

template<typename T, typename Func>
void tensor_reduce_op_kernel(
	T* output,		// [d1, d3]
	const T* input,	// [d1, d2, d3]
	int64_t d1,
	int64_t d2,
	int64_t d3,
	Func op
) {
	#pragma omp parallel for schedule(static) collapse(2)
	for (int64_t i = 0; i < d1; ++i) {
		for (int64_t j = 0; j < d3; ++j) {
			T tmp = input[i * d2 * d3 + j];
			for (int64_t k = 1; k < d2; ++k) {
				tmp = op(tmp, input[i * d2 * d3 + k * d3 + j]);
			}
			output[i * d3 + j] = tmp;
		}
	}
}

#define DEFINE_TENSOR_REDUCTION_OP(NAME, OP_LAMBDA) \
Tensor NAME(const Tensor &input, int axis) { \
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
	Tensor output(output_shape, input.dtype, input.device); \
	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype, tensor_reduce_op_kernel( \
		(T*) output.data_ptr(), \
		(const T*) input.data_ptr(), \
		d1, \
		d2, \
		d3, \
		OP_LAMBDA \
	)); \
	return output; \
}

DEFINE_TENSOR_REDUCTION_OP(tensor_reduction_sum, [](T a, T b) -> T {return a + b;})

}