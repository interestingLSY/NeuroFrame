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

template<typename T>
void broadcast_to_backward_kernel(
	T* __restrict__ input_grad,
	const T* __restrict__ output_grad,
	const std::vector<int64_t> &input_shape,
	const std::vector<int64_t> &target_shape
) {
	int64_t target_numel = get_product_over_vector(target_shape);
	for (int64_t target_index = 0; target_index < target_numel; ++target_index) {
		std::vector<int64_t> target_coords = get_coord_from_index(target_shape, target_index);
		std::vector<int64_t> input_coords = target_coords;
		for (size_t dim = 0; dim < input_shape.size(); ++dim) {
			input_coords[dim] %= input_shape[dim];
		}
		int64_t input_index = get_index_from_coord(input_shape, input_coords);
		input_grad[input_index] = input_grad[input_index] + output_grad[target_index];
	}
}

Tensor broadcast_to(const Tensor &input, const std::vector<int64_t> &target_shape) {
	std::vector<int64_t> input_shape = input.shape;
	if (input_shape.size() > target_shape.size()) {
		LOG_FATAL("Input shape (%s) has more dimensions than the target shape (%s)",
				  vec_to_string(input_shape).c_str(),
				  vec_to_string(target_shape).c_str());
	}
	// Add some leading 1s to the input shape
	while (input_shape.size() < target_shape.size()) {
		input_shape.insert(input_shape.begin(), 1);
	}

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

Tensor broadcast_to_backward(const Tensor &output_grad, const std::vector<int64_t> &original_input_shape) {
	std::vector<int64_t> target_shape = output_grad.shape;
	if (target_shape.size() < original_input_shape.size()) {
		LOG_FATAL("The out_grad shape (%s) has fewer dimensions than the input_grad shape (%s)",
				  vec_to_string(target_shape).c_str(),
				  vec_to_string(original_input_shape).c_str());
	}
	std::vector<int64_t> input_shape = original_input_shape;
	// Prepend some leading 1s to the input_shape to align with the target shape
	while (target_shape.size() > input_shape.size()) {
		input_shape.insert(input_shape.begin(), 1);
	}

	Tensor input_grad = Tensor::zeros(original_input_shape, output_grad.dtype, output_grad.device);
	DISPATCH_ON_DTYPE_CPU_BACKEND(output_grad.dtype,
		broadcast_to_backward_kernel(
			(T*) input_grad.data_ptr(),
			(const T*) output_grad.data_ptr(),
			input_shape,
			target_shape
		)
	);

	return input_grad;
}

}