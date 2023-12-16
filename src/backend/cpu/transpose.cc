#include "transpose.h"

#include <stdexcept>

#include "omp.h"

#include "src/tensor/tensor.h"
#include "utils.h"
#include "src/basic/log.h"
#include "src/utils/utils.h"

namespace NeuroFrame::Backend::CPU {

template<typename T>
void transpose_kernel(
	const T* input,
	T* output,
	int64_t d1,
	int64_t d2,
	int64_t d3,
	int64_t d4,
	int64_t d5
) {
	// We view the tensor `input` as [d1, d2, d3, d4, d5] and swap the 2-nd (d2) and 4-th (d4) axes.
	// The output tensor has shape [d1, d4, d3, d2, d5].
	#pragma omp parallel for collapse(3) schedule(static)
	for (int64_t i1 = 0; i1 < d1; ++i1) {
		for (int64_t i2 = 0; i2 < d2; ++i2) {
			for (int64_t i3 = 0; i3 < d3; ++i3) {
				for (int64_t i4 = 0; i4 < d4; ++i4) {
					for (int64_t i5 = 0; i5 < d5; ++i5) {
						output[i1 * d4 * d3 * d2 * d5 + i4 * d3 * d2 * d5 + i3 * d2 * d5 + i2 * d5 + i5] = input[i1 * d2 * d3 * d4 * d5 + i2 * d3 * d4 * d5 + i3 * d4 * d5 + i4 * d5 + i5];
					}
				}
			}
		}
	}
}

Tensor transpose(const Tensor &input, int axe1, int axe2) {
	int dim = input.dim();
	std::vector<int64_t> shape = input.shape;
	if (axe1 < 0 || axe1 >= dim || axe2 < 0 || axe2 >= dim) {
		LOG_FATAL("Transpose: Invalid axes");
	}
	if (axe1 == axe2) {
		return input;
	}
	if (axe1 > axe2) {
		std::swap(axe1, axe2);
	}

	int64_t d1 = get_product_over_vector(shape, 0, axe1);
	int64_t d2 = shape[axe1];
	int64_t d3 = get_product_over_vector(shape, axe1 + 1, axe2);
	int64_t d4 = shape[axe2];
	int64_t d5 = get_product_over_vector(shape, axe2 + 1);

	std::vector<int64_t> new_shape = shape;
	std::swap(new_shape[axe1], new_shape[axe2]);

	Tensor output(new_shape, input.dtype, input.device);
	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype, transpose_kernel(
		(const T*)input.data_ptr(),
		(T*)output.data_ptr(),
		d1, d2, d3, d4, d5
	));
	
	return output;
}

}