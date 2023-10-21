#include "transpose.h"

#include "src/backend/cpu/transpose.h"
#include "src/backend/cuda/transpose.h"

namespace NeuroFrame {

// transpose_forward_func: forward function for transpose
// Inputs:
//	- 0: Matrix A
// Outputs:
//	- 0: Matrix B = A^T
// Saved tensors: None
static op_forward_func_t transpose_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	if (a.dim() != 2) {
		LOG_FATAL("Transpose: Only support 2D tensors");
	}
	Tensor result = DISPATCH_TO_BACKEND(
		a.device.type,
		transpose(a)
	);
	return {result};
};

static op_backward_func_t transpose_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor result = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		transpose(output_grad[0])
	);
	return {result};
};

Tensor transpose_forward_manual(const Tensor &input, OpContext &ctx) {
	return transpose_forward_func({input}, ctx)[0];
}

Tensor transpose_backward_manual(const Tensor &output_grad, const OpContext &ctx) {
	return transpose_backward_func({output_grad}, ctx)[0];
}

Tensor transpose(const Tensor &input) {
	return perform_op(transpose_forward_func, transpose_backward_func, {input})[0];
}

}
