#include "relu.h"

#include <cuda_runtime.h>

#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

template<typename T>
__global__ void relu_forward_kernel(
	T* __restrict__ output,
	const T* __restrict__ input,
	int64_t n
) {
	#pragma unroll 4
	for (int64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
		output[i] = input[i] > (T)0.0 ? input[i] : (T)0.0;
}

Tensor relu_forward(const Tensor &input) {
	Tensor result(input.shape, input.dtype, input.device);
	int block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE;
	int grid_size = element_wise_kernel_get_num_grids(input.numel(), block_size);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype, relu_forward_kernel<<<grid_size, block_size>>>(
		(T*)result.data_ptr(),
		(const T*)input.data_ptr(),
		result.numel()
	));
	return result;
}

template<typename T>
__global__ void relu_backward_kernel(
	T* __restrict__ result_grad,
	const T* __restrict__ output_grad,
	const T* __restrict__ input,
	int64_t n
) {
	#pragma unroll 4
	for (int64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
		result_grad[i] = input[i] > (T)0.0 ? output_grad[i] : (T)0.0;
}

Tensor relu_backward(const Tensor &output_grad, const Tensor &input) {
	Tensor result_grad(input.shape, input.dtype, input.device);
	int block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE;
	int grid_size = element_wise_kernel_get_num_grids(input.numel(), block_size);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype, relu_backward_kernel<<<grid_size, block_size>>>(
		(T*)result_grad.data_ptr(),
		(const T*)output_grad.data_ptr(),
		(const T*)input.data_ptr(),
		result_grad.numel()
	));
	return result_grad;
}

}