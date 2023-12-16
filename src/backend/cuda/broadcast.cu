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
		int64_t target_index_copy = target_index;
		#pragma unroll 4
		for (int64_t dim_index = 0; dim_index < dim; dim_index++) {
			int64_t coord_on_this_dim = (target_index_copy / target_stride[dim_index]) % input_shape[dim_index];
			input_index = input_index * input_shape[dim_index] + coord_on_this_dim;
		}
		output[target_index_copy] = input[input_index];
	}
}

Tensor broadcast_to(const Tensor &input, const std::vector<int64_t> &target_shape) {
	if (input.dim() != target_shape.size()) {
		throw std::runtime_error("Input tensor and target shape must have the same number of dimensions.");
	}

	int64_t dim = input.dim();
	int64_t target_numel = get_product_over_vector(target_shape);

	std::vector<int64_t> input_shape_h = input.shape;
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
	int64_t grid_size = element_wise_kernel_get_num_grids(input.numel(), block_size);
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

}