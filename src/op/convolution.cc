#include "convolution.h"

#include <cassert>

#include "src/backend/cpu/convolution.h"
#include "src/backend/cuda/convolution.h"
#include "src/backend/cpu/matmul.h"
#include "src/backend/cuda/matmul.h"
#include "src/op/matmul.h"
#include "src/op/transpose.h"

namespace NeuroFrame {

// batched_convolution_forward_func: The forward function of convolution operator.
// Input:
//	- input: The input tensor, (batch_size, c_in, h, w)
//	- kernel: The kernel, (c_out, c_in, kh, kw)
// Output:
//	- result: The result tensor, (batch_size, c_out, h, w)
// SavedContext:
// 	- 0: input_img
// 	- 1: kernel
// OtherArgs: ConvolutionArgs

struct ConvolutionArgs {
	int64_t stride;
	int64_t dilation;
};

static op_forward_func_t batched_convolution_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	assert(other_args);

	ConvolutionArgs *args = (ConvolutionArgs*)other_args;

	Tensor input_img = input[0];
	Tensor kernel = input[1];

	if (input_img.dim() != 4) {
		LOG_FATAL("input.dim() != 4");
	}
	if (kernel.dim() != 4) {
		LOG_FATAL("kernel.dim() != 4");
	}
	if (input_img.shape[1] != kernel.shape[1]) {
		LOG_FATAL("input.shape[1] != kernel.shape[1]");
	}

	int64_t c_in = input_img.shape[1];
	int64_t kh = kernel.shape[2];
	int64_t kw = kernel.shape[3];
	if (kh % 2 == 0 || kw % 2 == 0) {
		LOG_FATAL("The kernel size must be odd");
	}
	if (c_in != kernel.shape[1]) {
		LOG_FATAL("The number of input channels does not match the number of kernel channels");
	}

	ctx.save_for_backward(input_img);
	ctx.save_for_backward(kernel);
	ctx.save_args(args, sizeof(ConvolutionArgs));

	Tensor real_result = DISPATCH_TO_BACKEND(
		input_img.device.type,
		convolution_forward(input_img, kernel, args->stride, args->dilation)
	);
	return {real_result};
};

// batched_convolution_backward_func
// Input:
//	- output_grad: The output gradient tensor, (batch_size, c_out, h, w)
// Output:
//	- input_img_grad: The input image gradient tensor, (batch_size, c_in, h, w)
//	- kernel_grad: The kernel gradient tensor, (c_out, c_in, kh, kw)
static op_backward_func_t batched_convolution_backward_func = [](const std::vector<Tensor> &output_grads, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grads, ctx);

	Tensor output_grad = output_grads[0];				// output_grad: (batch_size, c_out, h, w)
	Tensor input_img = ctx.get_saved_tensor(0);			// (b, c_in, kh, kw)
	Tensor kernel = ctx.get_saved_tensor(1);			// (c_out, c_in, kh, kw)
	ConvolutionArgs *args = (ConvolutionArgs*)ctx.get_saved_args();	// stride, dilation

	auto [input_img_grad, kernel_grad] = DISPATCH_TO_BACKEND(
		output_grad.device.type,
		convolution_backward(output_grad, input_img, kernel, args->stride, args->dilation)
	);
	
	return {input_img_grad, kernel_grad};
};

Tensor batched_convolution(const Tensor &input, const Tensor &kernel, int stride, int dilation) {
	ConvolutionArgs args = {stride, dilation};
	return perform_op(batched_convolution_forward_func, batched_convolution_backward_func, {input, kernel}, &args)[0];
}

}