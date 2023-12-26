#include "broadcast.h"

#include <stdexcept>

#include "omp.h"

#include "src/utils/utils.h"
#include "src/utils/cuda_utils.h"
#include "utils.h"
#include "reduction.cuh"

namespace NeuroFrame::Backend::CUDA {

// broadcast_to_kernel - Broadcast a tensor `input` to the given shape
template<typename T>
__global__ void broadcast_to_kernel(
	T* __restrict__ output,
	const T* __restrict__ input,
	const int64_t* __restrict__ input_shape,
	const int64_t* __restrict__ target_stride,
	const int64_t dim,
	const int64_t target_numel
) {
	for (int64_t target_index = blockIdx.x*blockDim.x + threadIdx.x; target_index < target_numel; target_index += blockDim.x*gridDim.x) {
		int64_t input_index = 0;
		#pragma unroll 4
		for (int64_t dim_index = 0; dim_index < dim; dim_index++) {
			int64_t coord_on_this_dim = (target_index / target_stride[dim_index]) % input_shape[dim_index];
			input_index = input_index * input_shape[dim_index] + coord_on_this_dim;
		}
		output[target_index] = input[input_index];
	}
}

template<typename T>
__global__ void broadcast_to_backward_kernel(
	T* __restrict__ input_grad,
	const T* __restrict__ output_grad,
	const int64_t* __restrict__ input_shape,
	const int64_t* __restrict__ input_stride,
	const int64_t* __restrict__ target_shape,
	const int64_t* __restrict__ target_stride,
	const int64_t dim,
	const int64_t input_numel,
	const int64_t target_numel
) {
	const int64_t group_id = blockIdx.x;
	const int64_t num_groups = gridDim.x;
	const int64_t worker_id = blockIdx.y*blockDim.x + threadIdx.x;
	const int64_t group_size = blockDim.x*gridDim.y;
	const int64_t numel_ratio = target_numel / input_numel;
	__shared__ int64_t s_input_shape[8], s_input_stride[8], s_target_shape[8], s_target_stride[8];
	if (threadIdx.x < dim) {
		s_input_shape[threadIdx.x] = input_shape[threadIdx.x];
		s_input_stride[threadIdx.x] = input_stride[threadIdx.x];
		s_target_shape[threadIdx.x] = target_shape[threadIdx.x];
		s_target_stride[threadIdx.x] = target_stride[threadIdx.x];
	}
	__syncthreads();
	// Iterate over the input tensor...
	for (int64_t input_index = group_id; input_index < input_numel; input_index += num_groups) {
		// Calculate the corresponding index in the output tensor
		int64_t target_base_index = 0;
		#pragma unroll 4
		for (int64_t dim_index = 0; dim_index < dim; dim_index++) {
			int64_t coord_on_this_dim = (input_index / s_input_stride[dim_index]) % s_input_shape[dim_index];
			target_base_index = target_base_index * s_target_shape[dim_index] + coord_on_this_dim;
		}
		T my_grad = 0;
		#pragma unroll 4
		for (int64_t target_offset = worker_id; target_offset < numel_ratio; target_offset += group_size){
			int64_t target_index = target_base_index;
			#pragma unroll 4
			for (int64_t dim_index = dim-1, temp_target_offset = target_offset; dim_index >= 0; --dim_index) {
				int64_t cur_dim_repeat = s_target_shape[dim_index] / s_input_shape[dim_index];
				int64_t cur_dim_repeat_index = temp_target_offset % cur_dim_repeat;
				temp_target_offset /= cur_dim_repeat;
				target_index += cur_dim_repeat_index * s_target_stride[dim_index];
			}
			my_grad += output_grad[target_index];
		}
		my_grad = block_reduce_sum(my_grad);
		if (worker_id == 0) {
			atomicAdd(&input_grad[input_index], my_grad);
		}
	}
}

Tensor broadcast_to(const Tensor &input, const std::vector<int64_t> &target_shape) {
	std::vector<int64_t> input_shape_h = input.shape;
	if (input_shape_h.size() > target_shape.size()) {
		LOG_FATAL("Input shape (%s) has more dimensions than the target shape (%s)",
				  vec_to_string(input_shape_h).c_str(),
				  vec_to_string(target_shape).c_str());
	}

	// Prepend some 1s to the input shape to make it the same number of dimensions as the target shape
	while (input_shape_h.size() < target_shape.size()) {
		input_shape_h.insert(input_shape_h.begin(), 1);
	}

	int64_t dim = input.dim();
	int64_t target_numel = get_product_over_vector(target_shape);
	std::vector<int64_t> target_stride_h = get_stride_from_shape(target_shape);

	static HostToDeviceAsyncCopyBuffer input_shape_buf, target_stride_buf;
	input_shape_buf.send(input_shape_h.data(), dim * sizeof(int64_t));
	target_stride_buf.send(target_stride_h.data(), dim * sizeof(int64_t));
	
	Tensor output(target_shape, input.dtype, input.device);
	int64_t block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE;
	int64_t grid_size = element_wise_kernel_get_num_grids(target_numel, block_size);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype, 
		broadcast_to_kernel<<<grid_size, block_size>>>(
			(T*) output.data_ptr(),
			(const T*) input.data_ptr(),
			(int64_t*) input_shape_buf.d_ptr,
			(int64_t*) target_stride_buf.d_ptr,
			dim,
			target_numel
		)
	);

	return output;
}

Tensor broadcast_to_backward(const Tensor &output_grad, const std::vector<int64_t> &original_input_shape_h) {
	std::vector<int64_t> target_shape = output_grad.shape;
	if (target_shape.size() < original_input_shape_h.size()) {
		LOG_FATAL("The out_grad shape (%s) has fewer dimensions than the input_grad shape (%s)",
				  vec_to_string(target_shape).c_str(),
				  vec_to_string(original_input_shape_h).c_str());
	}
	
	// Prepend some leading 1s to the input_shape to align with the target shape
	std::vector<int64_t> input_shape_h = original_input_shape_h;
	while (target_shape.size() > input_shape_h.size()) {
		input_shape_h.insert(input_shape_h.begin(), 1);
	}

	int64_t dim = output_grad.dim();
	int64_t input_numel = get_product_over_vector(input_shape_h);
	int64_t target_numel = output_grad.numel();
	std::vector<int64_t> input_stride_h = get_stride_from_shape(input_shape_h);
	std::vector<int64_t> target_stride_h = get_stride_from_shape(output_grad.shape);

	static HostToDeviceAsyncCopyBuffer input_shape_buf, input_stride_buf, target_shape_buf, target_stride_buf;
	input_shape_buf.send(input_shape_h.data(), dim * sizeof(int64_t));
	input_stride_buf.send(input_stride_h.data(), dim * sizeof(int64_t));
	target_shape_buf.send(target_shape.data(), dim * sizeof(int64_t));
	target_stride_buf.send(target_stride_h.data(), dim * sizeof(int64_t));

	Tensor input_grad = Tensor::zeros(original_input_shape_h, output_grad.dtype, output_grad.device);

	dim3 block_size, grid_size;
	if (target_numel/input_numel <= 1024L) {
		block_size = dim3(target_numel/input_numel, 1, 1);
		grid_size = dim3(std::min(input_numel, 16384L), 1, 1);
	} else {
		block_size = dim3(1024L, 1, 1);
		int64_t grid_y = target_numel/input_numel/1024L;
		grid_size = dim3(std::min(input_numel, 32768L/grid_y), grid_y, 1);
	}
	if (dim > 8) {
		LOG_FATAL("The number of dimensions (%ld) is too large. The current kernel only supports dim <= 8", dim);
	}
	DISPATCH_ON_DTYPE_CUDA_BACKEND(output_grad.dtype, 
		broadcast_to_backward_kernel<<<grid_size, block_size>>>(
			(T*) input_grad.data_ptr(),
			(const T*) output_grad.data_ptr(),
			(int64_t*) input_shape_buf.d_ptr,
			(int64_t*) input_stride_buf.d_ptr,
			(int64_t*) target_shape_buf.d_ptr,
			(int64_t*) target_stride_buf.d_ptr,
			dim,
			input_numel,
			target_numel
		)
	);

	return input_grad;
}

}