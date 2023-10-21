#include "pool.h"

#include <cmath>

#include "utils.h"

namespace NeuroFrame::Backend::CPU {

template<typename T>
void pool_forward_kernel(
	T* output,
	half* max_mask,
	const T* input,
	int64_t input_height,
	int64_t input_width,
	int64_t pool_size,
	int64_t batch_size
) {
	int64_t output_height = input_height / pool_size;
	int64_t output_width = input_width / pool_size;
	#pragma omp parallel for collapse(3) schedule(static)
	for (int64_t b = 0; b < batch_size; ++b) {
		for (int64_t i = 0; i < output_height; ++i) {
			for (int64_t j = 0; j < output_width; ++j) {
				const T* cur_pool = input + b*input_height*input_width + i*pool_size*input_width + j*pool_size;
				T max_value = cur_pool[0];
				int64_t max_index = b*input_height*input_width + i*pool_size*input_width + j*pool_size;	
				for (int64_t k = 0; k < pool_size; ++k) {
					for (int64_t l = 0; l < pool_size; ++l) {
						int64_t index = b*input_height*input_width + (i*pool_size+k)*input_width + j*pool_size+l;
						if (input[index] > max_value) {
							max_value = input[index];
							max_index = index;
						}
					}
				}
				output[b*output_height*output_width + i*output_width + j] = max_value;
				max_mask[max_index] = 1.0f;
			}
		}
	}
}

std::pair<Tensor, Tensor> pool_forward(const Tensor &input, int pool_size) {
	if (input.dim() < 2) {
		LOG_FATAL("Input tensor must have at least 2 dimensions");
	}

	int64_t input_height = input.shape[input.shape.size()-2];
	int64_t input_width = input.shape[input.shape.size()-1];
	if (input_height % pool_size != 0 || input_width % pool_size != 0) {
		LOG_FATAL("Input tensor's height and width must be divisible by pool_size");
	}

	std::vector<int64_t> output_shape;
	int64_t batch_size = 1;
	for (int i = 0; i < (int)input.shape.size()-2; ++i) {
		output_shape.push_back(input.shape[i]);
		batch_size *= input.shape[i];
	}
	output_shape.push_back(input_height/pool_size);
	output_shape.push_back(input_width/pool_size);
	Tensor output(output_shape, input.dtype, input.device);
	Tensor max_mask = Tensor::zeros(input.shape, dtype_t::FLOAT16, input.device);

	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype, pool_forward_kernel(
		(T*) output.data_ptr(),
		(half*) max_mask.data_ptr(),
		(const T*) input.data_ptr(),
		input_height,
		input_width,
		pool_size,
		batch_size
	));

	return {output, max_mask};
}

template<typename T>
void pool_backward_kernel(
	T* input_grad,
	const T* output_grad,
	const half* max_mask,
	int64_t input_height,
	int64_t input_width,
	int64_t pool_size,
	int64_t batch_size
) {
	int64_t output_height = input_height / pool_size;
	int64_t output_width = input_width / pool_size;
	#pragma omp parallel for collapse(3) schedule(static)
	for (int64_t b = 0; b < batch_size; ++b) {
		for (int64_t i = 0; i < output_height; ++i) {
			for (int64_t j = 0; j < output_width; ++j) {
				T cur_pool_output_grad = output_grad[b*output_height*output_width + i*output_width + j];
				for (int64_t k = 0; k < pool_size; ++k) {
					for (int64_t l = 0; l < pool_size; ++l) {
						int64_t index = b*input_height*input_width + (i*pool_size+k)*input_width + j*pool_size+l;
						input_grad[index] = cur_pool_output_grad * max_mask[index];
					}
				}
			}
		}
	}
}

Tensor pool_backward(const Tensor &output_grad, const Tensor &max_mask, int pool_size) {
	int64_t input_height = output_grad.shape[output_grad.shape.size()-2] * pool_size;
	int64_t input_width = output_grad.shape[output_grad.shape.size()-1] * pool_size;

	std::vector<int64_t> input_grad_shape;
	int64_t batch_size = 1;
	for (int i = 0; i < (int)output_grad.shape.size()-2; ++i) {
		input_grad_shape.push_back(output_grad.shape[i]);
		batch_size *= output_grad.shape[i];
	}
	input_grad_shape.push_back(input_height);
	input_grad_shape.push_back(input_width);
	Tensor input_grad(input_grad_shape, output_grad.dtype, output_grad.device);

	DISPATCH_ON_DTYPE_CPU_BACKEND(output_grad.dtype, pool_backward_kernel(
		(T*) input_grad.data_ptr(),
		(const T*) output_grad.data_ptr(),
		(const half*) max_mask.data_ptr(),
		input_height,
		input_width,
		pool_size,
		batch_size
	));

	return input_grad;
}

}

