#include "relu.h"

#include "utils.h"

namespace NeuroFrame::Backend::CPU {

template<typename T>
void relu_forward_kernel(
	T* output,
	const T* input,
	int64_t n
) {
	for (int64_t i = 0; i < n; i++) {
		output[i] = input[i] > (T)0.0 ? input[i] : (T)0.0;
	}
}

Tensor relu_forward(const Tensor &input) {
	Tensor result(input.shape, input.dtype, input.device);
	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype, relu_forward_kernel(
		(T*)result.data_ptr(),
		(const T*)input.data_ptr(),
		result.numel()
	));
	return result;
}

template<typename T>
void relu_backward_kernel(
	T* result_grad,
	const T* output_grad,
	const T* input,
	int64_t n
) {
	for (int64_t i = 0; i < n; i++) {
		result_grad[i] = input[i] > (T)0.0 ? output_grad[i] : (T)0.0;
	}
}

Tensor relu_backward(const Tensor &output_grad, const Tensor &input) {
	Tensor result_grad(input.shape, input.dtype, input.device);
	DISPATCH_ON_DTYPE_CPU_BACKEND(input.dtype, relu_backward_kernel(
		(T*)result_grad.data_ptr(),
		(const T*)output_grad.data_ptr(),
		(const T*)input.data_ptr(),
		result_grad.numel()
	));
	return result_grad;
}

}