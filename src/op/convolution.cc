#include "convolution.h"

#include "src/backend/cpu/convolution.h"
#include "src/backend/cuda/convolution.h"
#include "src/backend/cpu/matmul.h"
#include "src/backend/cuda/matmul.h"
#include "src/op/matmul.h"

namespace NeuroFrame {

// batched_convolution_forward_func: The forward function of convolution operator.
// Input:
//	- input: The input tensor, (batch_size, c_in, h, w)
//	- kernel: The kernel, (c_out, c_in, kh, kw)
// Output:
//	- result: The result tensor, (batch_size, c_out, h, w)
// SavedContext:
// 	- 0: kernel
// 	- 1: im2col_result
// OtherArgs: None
static op_forward_func_t batched_convolution_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);

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

	int64_t batch_size = input_img.shape[0];
	int64_t c_in = input_img.shape[1];
	int64_t h = input_img.shape[2];
	int64_t w = input_img.shape[3];
	int64_t c_out = kernel.shape[0];
	int64_t kh = kernel.shape[2];
	int64_t kw = kernel.shape[3];
	if (kh % 2 == 0 || kw % 2 == 0) {
		LOG_FATAL("The kernel size must be odd");
	}
	if (c_in != kernel.shape[1]) {
		LOG_FATAL("The number of input channels does not match the number of kernel channels");
	}

	Tensor im2col_result = DISPATCH_TO_BACKEND(
		input_img.device.type,
		batched_im2col(input_img, kh, kw)
	);	// (batch_size, h*w, c_in*kh*kw)

	// Here we want to calculate (im2col_result * (kernel^T))^T, so we calculate kernel * im2col_result^T instead.
	Tensor conv_result = DISPATCH_TO_BACKEND(
		input_img.device.type,
		matmul(
			kernel.reshape({c_out, c_in*kh*kw}),
			im2col_result.reshape({batch_size*h*w, c_in*kh*kw}),
			false, true
		)
	).reshape({batch_size, c_out, h, w});	// (batch_size, c_out, h*w)

	ctx.save_for_backward(kernel);
	ctx.save_for_backward(im2col_result);

	return {conv_result};
};

// batched_convolution_backward_func
// Input:
//	- output_grad: The output gradient tensor, (batch_size, c_out, h, w)
// Output:
//	- input_img_grad: The input image gradient tensor, (batch_size, c_in, h, w)
//	- kernel_grad: The kernel gradient tensor, (batch_size, c_out, c_in, kh, kw)
static op_backward_func_t batched_convolution_backward_func = [](const std::vector<Tensor> &output_grads, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grads, ctx);

	Tensor output_grad = output_grads[0];				// output_grad: (batch_size, c_out, h, w)
	Tensor kernel = ctx.get_saved_tensors()[0];			// (c_out, c_in, kh, kw)
	Tensor im2col_result = ctx.get_saved_tensors()[1];	// (batch_size, h*w, c_in*kh*kw)
	int64_t batch_size = im2col_result.shape[0];
	int64_t h_mult_w = im2col_result.shape[1];
	int64_t c_in_mult_kh_mult_kw = im2col_result.shape[2];
	int64_t c_out = kernel.shape[0];
	int64_t c_in = kernel.shape[1];
	int64_t kh = kernel.shape[2];
	int64_t kw = kernel.shape[3];
	int64_t h = output_grad.shape[2];
	int64_t w = output_grad.shape[3];

	// output = kernel * im2col_result^T
	// => im2col_result' = (kernel^T * output')^T = output'^T * kernel
	// => kernel' = output' * im2col_result

	Tensor im2col_result_grad = DISPATCH_TO_BACKEND(
		im2col_result.device.type,
		batched_matmul(
			output_grad.reshape({batch_size, c_out, h_mult_w}),
			kernel.reshape({c_out, c_in_mult_kh_mult_kw}),
			true, false
		)
	).reshape({batch_size, h_mult_w, c_in_mult_kh_mult_kw});	// (batch_size, h*w, c_in*kh*kw)

	Tensor kernel_grad = DISPATCH_TO_BACKEND(
		im2col_result.device.type,
		batched_matmul(
			output_grad.reshape({batch_size, c_out, h_mult_w}),
			im2col_result.reshape({batch_size, h_mult_w, c_in_mult_kh_mult_kw}),
			false, false
		)
	).reshape({batch_size, c_in, c_out, kh, kw});

	Tensor input_img_grad = DISPATCH_TO_BACKEND(
		im2col_result.device.type,
		batched_col2im(im2col_result_grad, c_in, h, w, kh, kw)
	);	// (batch_size, c_in, h, w)
	
	return {input_img_grad, kernel_grad};
};

Tensor batched_convolution_forward_manual(const Tensor &input_img, const Tensor &kernel, OpContext &ctx) {
	return batched_convolution_forward_func({input_img, kernel}, ctx, nullptr)[0];
}

std::vector<Tensor> batched_convolution_backward_manual(const Tensor &output_grad, const OpContext &ctx) {
	return batched_convolution_backward_func({output_grad}, ctx);
}

Tensor batched_convolution(const Tensor &input, const Tensor &kernel) {
	return perform_op(batched_convolution_forward_func, batched_convolution_backward_func, {input, kernel})[0];
}

}