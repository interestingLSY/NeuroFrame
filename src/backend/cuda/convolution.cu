#include "convolution.h"

#include <cudnn.h>

#include "src/basic/log.h"
#include "src/tensor/tensor.h"

#include "utils.h"
#include "cudnn_utils.h"

namespace NeuroFrame::Backend::CUDA {

Tensor convolution_forward(const Tensor &input_img, const Tensor &kernel, const int64_t stride, const int64_t dilation) {
	if (input_img.dim() != 4) {
		LOG_FATAL("input_img.dim() != 4");
	}
	if (kernel.dim() != 4) {
		LOG_FATAL("kernel.dim() != 4");
	}
	const int64_t b = input_img.shape[0];
	const int64_t c_in = input_img.shape[1];
	const int64_t h = input_img.shape[2];
	const int64_t w = input_img.shape[3];
	const int64_t c_out = kernel.shape[0];
	const int64_t kh = kernel.shape[2];
	const int64_t kw = kernel.shape[3];

	const int64_t out_h = (h + 2*(kh/2) - (dilation*(kh-1)+1)) / stride + 1;
	const int64_t out_w = (w + 2*(kw/2) - (dilation*(kw-1)+1)) / stride + 1;

	Tensor result({b, c_out, out_h, out_w}, input_img.dtype, input_img.device);

	auto [alpha_ptr, beta_ptr] = get_alpha_beta_ptrs(input_img.dtype);

	static cudnnTensorDescriptor_t input_desc;
	static cudnnTensorDescriptor_t output_desc;
	static cudnnFilterDescriptor_t kernel_desc;
	static cudnnConvolutionDescriptor_t conv_desc;
	static bool desc_created = false;
	if (!desc_created) {
		desc_created = true;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
		CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_desc));
		CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
	}

	cudnnDataType_t cudnn_datatype = get_cudnn_data_type(input_img.dtype);
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, cudnn_datatype, b, c_in, h, w));
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, cudnn_datatype, b, c_out, out_h, out_w));
	CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_desc, cudnn_datatype, CUDNN_TENSOR_NCHW, c_out, c_in, kh, kw));
	CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, kh/2, kw/2, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, cudnn_datatype));

	cudnnConvolutionFwdAlgoPerf_t perf_result; int _returned_algo_count;
	CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
		cudnn_handle,
		input_desc,
		kernel_desc,
		conv_desc,
		output_desc,
		1,
		&_returned_algo_count,
		&perf_result
	));
	size_t workspace_bytes = perf_result.memory;
	void* workspace_ptr;
	CUDA_CHECK(cudaMallocAsync(&workspace_ptr, workspace_bytes, cudaStreamDefault));

	CUDNN_CHECK(cudnnConvolutionForward(
		cudnn_handle,
		alpha_ptr,
		input_desc,
		input_img.data_ptr(),
		kernel_desc,
		kernel.data_ptr(),
		conv_desc,
		perf_result.algo,
		workspace_ptr,
		workspace_bytes,
		beta_ptr,
		output_desc,
		result.data_ptr()
	));

	CUDA_CHECK(cudaFreeAsync(workspace_ptr, cudaStreamDefault));
	return result;
}

std::tuple<Tensor, Tensor> convolution_backward(const Tensor &output_grad, const Tensor &input_img, const Tensor &kernel, const int64_t stride, const int64_t dilation) {
	const int64_t b = input_img.shape[0];
	const int64_t c_out = output_grad.shape[1];
	const int64_t h = input_img.shape[2];
	const int64_t w = input_img.shape[3];
	const int64_t c_in = kernel.shape[1];
	const int64_t kh = kernel.shape[2];
	const int64_t kw = kernel.shape[3];
	const int64_t out_h = output_grad.shape[2];
	const int64_t out_w = output_grad.shape[3];

	Tensor input_grad({b, c_in, h, w}, output_grad.dtype, output_grad.device);
	Tensor kernel_grad({c_out, c_in, kh, kw}, output_grad.dtype, output_grad.device);

	// Get alpha and beta
	auto [alpha_ptr, beta_ptr] = get_alpha_beta_ptrs(input_img.dtype);

	static cudnnTensorDescriptor_t input_grad_desc;
	static cudnnTensorDescriptor_t output_grad_desc;
	static cudnnFilterDescriptor_t kernel_desc;
	static cudnnConvolutionDescriptor_t conv_desc;
	static bool desc_created = false;
	if (!desc_created) {
		desc_created = true;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_grad_desc));
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_grad_desc));
		CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_desc));
		CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
	}

	cudnnDataType_t cudnn_datatype = get_cudnn_data_type(output_grad.dtype);
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_grad_desc, CUDNN_TENSOR_NCHW, cudnn_datatype, b, c_in, h, w));
	CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_grad_desc, CUDNN_TENSOR_NCHW, cudnn_datatype, b, c_out, out_h, out_w));
	CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_desc, cudnn_datatype, CUDNN_TENSOR_NCHW, c_out, c_in, kh, kw));
	CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, kh/2, kw/2, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, cudnn_datatype));

	cudnnConvolutionBwdDataAlgoPerf_t data_perf_result; int _returned_algo_count;
	CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
		cudnn_handle,
		kernel_desc,
		output_grad_desc,
		conv_desc,
		input_grad_desc,
		1,
		&_returned_algo_count,
		&data_perf_result
	));
	cudnnConvolutionBwdFilterAlgoPerf_t filter_perf_result;
	CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
		cudnn_handle,
		input_grad_desc,
		output_grad_desc,
		conv_desc,
		kernel_desc,
		1,
		&_returned_algo_count,
		&filter_perf_result
	));

	size_t workspace_bytes = std::max(data_perf_result.memory, filter_perf_result.memory);
	void* workspace_ptr;
	CUDA_CHECK(cudaMallocAsync(&workspace_ptr, workspace_bytes, cudaStreamDefault));

	CUDNN_CHECK(cudnnConvolutionBackwardData(
		cudnn_handle,
		alpha_ptr,
		kernel_desc,
		kernel.data_ptr(),
		output_grad_desc,
		output_grad.data_ptr(),
		conv_desc,
		data_perf_result.algo,
		workspace_ptr,
		workspace_bytes,
		beta_ptr,
		input_grad_desc,
		input_grad.data_ptr()
	));
	
	CUDNN_CHECK(cudnnConvolutionBackwardFilter(
		cudnn_handle,
		alpha_ptr,
		input_grad_desc,
		input_img.data_ptr(),
		output_grad_desc,
		output_grad.data_ptr(),
		conv_desc,
		filter_perf_result.algo,
		workspace_ptr,
		workspace_bytes,
		beta_ptr,
		kernel_desc,
		kernel_grad.data_ptr()
	));

	CUDA_CHECK(cudaFreeAsync(workspace_ptr, cudaStreamDefault));
	return {input_grad, kernel_grad};
}


}