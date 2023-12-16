#include "broadcast.h"

#include <stdexcept>

#include "omp.h"

#include "src/utils/utils.h"
#include "utils.h"

namespace NeuroFrame::Backend::CPU {

template<typename T>
void broadcast_to_kernel(
	T* __restrict__ output,
	const T* __restrict__ input,
	const std::vector<int64_t> &input_shape,
	const std::vector<int64_t> &target_shape
) {
	int64_t target_numel = get_product_over_vector(target_shape);
	#pragma omp parallel for schedule(static)
	for (int64_t target_index = 0; target_index < target_numel; ++target_index) {
		std::vector<int64_t> target_coords = get_coord_from_index(target_shape, target_index);
		std::vector<int64_t> input_coords = target_coords;
		for (size_t dim = 0; dim < input_shape.size(); ++dim) {
			input_coords[dim] %= input_shape[dim];
		}
		int64_t input_index = get_index_from_coord(input_shape, input_coords);
		output[target_index] = input[input_index];
	}
}

Tensor broadcast_to(const Tensor &input, const std::vector<int64_t> &target_shape) {
	if (input.dim() != (int64_t)target_shape.size()) {
		throw std::runtime_error("Input tensor and target shape must have the same number of dimensions.");
	}

	std::vector<int64_t> input_shape = input.shape;

	Tensor output(target_shape, input.dtype, input.device);
	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype,
		broadcast_to_kernel(
			(T*) output.data_ptr(),
			(const T*) input.data_ptr(),
			input_shape,
			target_shape
		)
	);

	return output;
}

}