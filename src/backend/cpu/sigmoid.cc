#include "sigmoid.h"

#include <cmath>

#include "utils.h"

namespace NeuroFrame::Backend::CPU {

template<typename T>
void sigmoid_forward_kernel(
	T* output,
	const T* input,
	int64_t n
) {
	for (int64_t i = 0; i < n; i++) {
		output[i] = 1.0 / (1.0 + std::exp(-input[i]));
	}
}

Tensor sigmoid_forward(const Tensor &input) {
	Tensor result(input.shape, input.dtype, input.device);
	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype, sigmoid_forward_kernel(
		(T*)result.data_ptr(),
		(const T*)input.data_ptr(),
		result.numel()
	));
	return result;
}

template<typename T>
void sigmoid_backward_kernel(
	T* result_grad,
	const T* output_grad,
	const T* output,
	int64_t n
) {
	for (int64_t i = 0; i < n; i++) {
		result_grad[i] = output_grad[i] * output[i] * (1.0 - output[i]);
	}
}

Tensor sigmoid_backward(const Tensor &output_grad, const Tensor &output) {
	Tensor result_grad(output.shape, output.dtype, output.device);
	DISPATCH_ON_DTYPE_CPU_BACKEND(output.dtype, sigmoid_backward_kernel(
		(T*)result_grad.data_ptr(),
		(const T*)output_grad.data_ptr(),
		(const T*)output.data_ptr(),
		result_grad.numel()
	));
	return result_grad;
}

}