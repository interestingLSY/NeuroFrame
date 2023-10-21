#include "pool.h"

#include <stdexcept>
#include <cuda_runtime.h>

#include "src/tensor/tensor.h"
#include "src/basic/log.h"
#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

// pooling kernel
// Grid size: [batch_size, input_height/pool_size]
// Block size: [min(input_width/pool_size, 512)]
// Each thread is responsible for one block
template<typename T>
__global__ void pool_kernel(
	T* output,			// [batch_size, input_height/pool_size, input_width/pool_size]
	half* max_mask,	// [batch_size, input_height, input_width], Use half* since our tensor currently does not support bool
	const T* input,		// [batch_size, input_height, input_width]
	int64_t input_height,
	int64_t input_width,
	int64_t pool_size,
	int64_t batch_size
) {
	int64_t batch_index = blockIdx.x;
	int64_t pool_y_index = blockIdx.y;
	for (int64_t pool_x_index = threadIdx.x; pool_x_index < input_width/pool_size; pool_x_index += blockDim.x) {
		int64_t max_offset_x = 0, max_offset_y = 0;
		const T* cur_pool = input
			+ batch_index*input_width*input_height
			+ pool_y_index*pool_size*input_width
			+ pool_x_index*pool_size;
		T cur_max = cur_pool[0];
		for (int64_t i = 0; i < pool_size; ++i) {
			for (int64_t j = 0; j < pool_size; ++j) {
				T cur = cur_pool[i*input_width + j];
				if (cur > cur_max) {
					cur_max = cur;
					max_offset_x = j;
					max_offset_y = i;
				}
			}
		}
		// printf("%f\n", (double)cur_max);
		output[
			batch_index*(input_width/pool_size)*(input_height/pool_size)
			+ pool_y_index*(input_width/pool_size)
			+ pool_x_index] = cur_max;
		max_mask[
			batch_index*input_width*input_height
			+ (pool_y_index*pool_size + max_offset_y)*input_width
			+ pool_x_index*pool_size + max_offset_x] = (half)1.0;
	}
}

// Return value: (output, max_mask)
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
	output_shape.push_back(input.shape[input.shape.size()-2]/pool_size);
	output_shape.push_back(input.shape[input.shape.size()-1]/pool_size);
	Tensor output(output_shape, input.dtype, input.device);
	Tensor max_mask = Tensor::zeros(input.shape, dtype_t::FLOAT16, input.device);

	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype, pool_kernel<<<dim3(batch_size, input_height/pool_size), std::min(input_width/pool_size, (int64_t)512)>>>(
		(T*) output.data_ptr(),
		(half*) max_mask.data_ptr(),
		(const T*) input.data_ptr(),
		input_height,
		input_width,
		pool_size,
		batch_size
	));

	return std::make_pair(output, max_mask);
}

// pooling (backward) kernel
// Grid size: [batch_size, input_height/pool_size]
// Block size: [min(input_width/pool_size, 512)]
// Each thread is responsible for one block
template<typename T>
__global__ void pool_backward_kernel(
	T* input_grad,
	const T* output_grad,
	const half* max_mask,
	int64_t input_height,
	int64_t input_width,
	int64_t pool_size,
	int64_t batch_size
) {
	int64_t batch_index = blockIdx.x;
	int64_t pool_y_index = blockIdx.y;
	for (int64_t pool_x_index = threadIdx.x; pool_x_index < input_width/pool_size; pool_x_index += blockDim.x) {
		T* cur_pool = input_grad
			+ batch_index*input_width*input_height
			+ pool_y_index*pool_size*input_width
			+ pool_x_index*pool_size;
		T cur_output_grad = output_grad[
			+ batch_index*(input_width/pool_size)*(input_height/pool_size)
			+ pool_y_index*(input_width/pool_size)
			+ pool_x_index];
		const half* cur_max_mask = max_mask
			+ batch_index*input_width*input_height
			+ pool_y_index*pool_size*input_width
			+ pool_x_index*pool_size;
		for (int64_t i = 0; i < pool_size; ++i) {
			for (int64_t j = 0; j < pool_size; ++j) {
				cur_pool[i*input_width + j] = cur_output_grad * 
					((cur_max_mask[i*input_width + j] > (half)0.5) ? (T)1.0 : (T)0.0);
			}
		}
	}
}

Tensor pool_backward(const Tensor &output_grad, const Tensor &max_mask, int pool_size) {
	int64_t input_height = output_grad.shape[output_grad.shape.size()-2]*pool_size;
	int64_t input_width = output_grad.shape[output_grad.shape.size()-1]*pool_size;

	std::vector<int64_t> input_grad_shape;
	int64_t batch_size = 1;
	for (int i = 0; i < (int)output_grad.shape.size()-2; ++i) {
		input_grad_shape.push_back(output_grad.shape[i]);
		batch_size *= output_grad.shape[i];
	}
	input_grad_shape.push_back(input_height);
	input_grad_shape.push_back(input_width);
	Tensor input_grad(input_grad_shape, output_grad.dtype, output_grad.device);

	DISPATCH_ON_DTYPE_CUDA_BACKEND(output_grad.dtype, pool_backward_kernel<<<dim3(batch_size, input_height/pool_size), std::min(input_width/pool_size, (int64_t)512)>>>(
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