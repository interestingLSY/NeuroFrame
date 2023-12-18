#include "broadcast.h"

#include <stdexcept>

#include "omp.h"

#include "src/utils/utils.h"
#include "src/utils/cuda_utils.h"
#include "utils.h"

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
		atomicAdd(input_grad + input_index, output_grad[target_index]);
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

	static int64_t *input_shape_d = nullptr, *target_stride_d = nullptr, dim_d = 0;
	if (dim_d < dim) {
		if (input_shape_d != nullptr) {
			CUDA_CHECK(cudaFree(input_shape_d));
			CUDA_CHECK(cudaFree(target_stride_d));
		}
		while (dim_d < dim) {
			dim_d = dim_d == 0 ? 16 : dim_d * 2;
		}
		CUDA_CHECK(cudaMalloc(&input_shape_d, dim_d * sizeof(int64_t)));
		CUDA_CHECK(cudaMalloc(&target_stride_d, dim_d * sizeof(int64_t)));
	}
	CUDA_CHECK(cudaMemcpy(input_shape_d, input_shape_h.data(), dim * sizeof(int64_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(target_stride_d, target_stride_h.data(), dim * sizeof(int64_t), cudaMemcpyHostToDevice));

	Tensor output(target_shape, input.dtype, input.device);
	int64_t block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE;
	int64_t grid_size = element_wise_kernel_get_num_grids(target_numel, block_size);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype, 
		broadcast_to_kernel<<<grid_size, block_size>>>(
			(T*) output.data_ptr(),
			(const T*) input.data_ptr(),
			input_shape_d,
			target_stride_d,
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
	int64_t target_numel = output_grad.numel();
	std::vector<int64_t> target_stride_h = get_stride_from_shape(output_grad.shape);

	static int64_t *input_shape_d = nullptr, *target_stride_d = nullptr, dim_d = 0;
	if (dim_d < dim) {
		if (input_shape_d != nullptr) {
			CUDA_CHECK(cudaFree(input_shape_d));
			CUDA_CHECK(cudaFree(target_stride_d));
		}
		while (dim_d < dim) {
			dim_d = dim_d == 0 ? 16 : dim_d * 2;
		}
		CUDA_CHECK(cudaMalloc(&input_shape_d, dim_d * sizeof(int64_t)));
		CUDA_CHECK(cudaMalloc(&target_stride_d, dim_d * sizeof(int64_t)));
	}
	CUDA_CHECK(cudaMemcpy(input_shape_d, input_shape_h.data(), dim * sizeof(int64_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(target_stride_d, target_stride_h.data(), dim * sizeof(int64_t), cudaMemcpyHostToDevice));

	Tensor input_grad = Tensor::zeros(original_input_shape_h, output_grad.dtype, output_grad.device);
	int64_t block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE;
	int64_t grid_size = element_wise_kernel_get_num_grids(target_numel, block_size);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(output_grad.dtype, 
		broadcast_to_backward_kernel<<<grid_size, block_size>>>(
			(T*) input_grad.data_ptr(),
			(const T*) output_grad.data_ptr(),
			input_shape_d,
			target_stride_d,
			dim,
			target_numel
		)
	);

	return input_grad;
}

}