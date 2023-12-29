#include "pool.h"

#include <stdexcept>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "src/tensor/tensor.h"
#include "src/basic/log.h"
#include "utils.h"
#include "cudnn_utils.h"

namespace NeuroFrame::Backend::CUDA {

Tensor pool_forward(const Tensor &input, int pool_size, int stride, int padding) {
	if (input.dim() != 4) {
		LOG_FATAL("Input tensor must have 4 dimensions");
	}

	int64_t n = input.shape[0];
	int64_t c = input.shape[1];
	int64_t h = input.shape[2];
	int64_t w = input.shape[3];
	int out_n, out_c, out_h, out_w;
	
	static cudnnPoolingDescriptor_t pool_desc;
	static cudnnTensorDescriptor_t input_desc, output_desc;
	static bool desc_inited = false;
	if (!desc_inited) {
		CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
		desc_inited = true;
	}
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(
		input_desc,
		CUDNN_TENSOR_NCHW,
		get_cudnn_data_type(input.dtype),
		n,
		c,
		h,
		w
	));
	CUDNN_CHECK(cudnnSetPooling2dDescriptor(
		pool_desc,
		CUDNN_POOLING_MAX,
		CUDNN_NOT_PROPAGATE_NAN,
		pool_size,
		pool_size,
		padding,
		padding,
		stride,
		stride
	));
	CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(
		pool_desc,
		input_desc,
		&out_n,
		&out_c,
		&out_h,
		&out_w
	));
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(
		output_desc,
		CUDNN_TENSOR_NCHW,
		get_cudnn_data_type(input.dtype),
		out_n,
		out_c,
		out_h,
		out_w
	));
	assert(out_n == n);
	assert(out_c == c);

	std::vector<int64_t> output_shape = {n, c, out_h, out_w};
	Tensor output(output_shape, input.dtype, input.device);

	auto [alpha_ptr, beta_ptr] = get_alpha_beta_ptrs(input.dtype);

	CUDNN_CHECK(cudnnPoolingForward(
		cudnn_handle,
		pool_desc,
		alpha_ptr,
		input_desc,
		input.data_ptr(),
		beta_ptr,
		output_desc,
		output.data_ptr()
	));

	return output;
}

Tensor pool_backward(const Tensor &output_grad, const Tensor &input, const Tensor &output, int pool_size, int stride, int padding) {
	static cudnnPoolingDescriptor_t pool_desc;
	static cudnnTensorDescriptor_t input_desc, output_desc;	// input_desc and input_grad_desc are the same thing
	static bool desc_inited = false;
	if (!desc_inited) {
		CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
		desc_inited = true;
	}

	int64_t n = input.shape[0];
	int64_t c = input.shape[1];
	int64_t h = input.shape[2];
	int64_t w = input.shape[3];
	int64_t out_h = output.shape[2];
	int64_t out_w = output.shape[3];

	CUDNN_CHECK(cudnnSetTensor4dDescriptor(
		input_desc,
		CUDNN_TENSOR_NCHW,
		get_cudnn_data_type(input.dtype),
		n, c, h, w
	));
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(
		output_desc,
		CUDNN_TENSOR_NCHW,
		get_cudnn_data_type(output.dtype),
		n, c, out_h, out_w
	));
	CUDNN_CHECK(cudnnSetPooling2dDescriptor(
		pool_desc,
		CUDNN_POOLING_MAX,
		CUDNN_NOT_PROPAGATE_NAN,
		pool_size,
		pool_size,
		padding,
		padding,
		stride,
		stride
	));

	std::vector<int64_t> input_grad_shape = {n, c, h, w};
	Tensor input_grad(input_grad_shape, input.dtype, input.device);

	auto [alpha_ptr, beta_ptr] = get_alpha_beta_ptrs(input.dtype);

	CUDNN_CHECK(cudnnPoolingBackward(
		cudnn_handle,
		pool_desc,
		alpha_ptr,
		output_desc,
		output.data_ptr(),
		output_desc,
		output_grad.data_ptr(),
		input_desc,
		input.data_ptr(),
		beta_ptr,
		input_desc,
		input_grad.data_ptr()
	));

	return input_grad;
}

}