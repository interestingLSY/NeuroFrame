#include "relu.h"

#include <cuda_runtime.h>

#include "utils.h"

namespace NeuroFrame::Backend::CUDA {

template<typename T>
__global__ void sigmoid_forward_kernel(
	T* __restrict__ output,
	const T* __restrict__ input,
	int64_t n
) {
	#pragma unroll 4
	for (int64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
		output[i] = (T)(1.0f / (1.0f + __expf((float)-input[i])));
}

Tensor sigmoid_forward(const Tensor &input) {
	Tensor result(input.shape, input.dtype, input.device);
	int block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE;
	int grid_size = element_wise_kernel_get_num_grids(input.numel(), block_size);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(input.dtype, sigmoid_forward_kernel<<<grid_size, block_size>>>(
		(T*)result.data_ptr(),
		(const T*)input.data_ptr(),
		result.numel()
	));
	return result;
}

template<typename T>
__global__ void sigmoid_backward_kernel(
	T* __restrict__ result_grad,
	const T* __restrict__ output_grad,
	const T* __restrict__ output,
	int64_t n
) {
	#pragma unroll 4
	for (int64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x)
		result_grad[i] = output_grad[i] * output[i] * ((T)1.0 - output[i]);
}

Tensor sigmoid_backward(const Tensor &output_grad, const Tensor &output) {
	Tensor result_grad(output.shape, output.dtype, output.device);
	int block_size = ELEMENT_WISE_KERNEL_BLOCK_SIZE;
	int grid_size = element_wise_kernel_get_num_grids(output.numel(), block_size);
	DISPATCH_ON_DTYPE_CUDA_BACKEND(output.dtype, sigmoid_backward_kernel<<<grid_size, block_size>>>(
		(T*)result_grad.data_ptr(),
		(const T*)output_grad.data_ptr(),
		(const T*)output.data_ptr(),
		result_grad.numel()
	));
	return result_grad;
}

}