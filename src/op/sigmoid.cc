#include "sigmoid.h"

#include "src/backend/cpu/sigmoid.h"
#include "src/backend/cuda/sigmoid.h"

namespace NeuroFrame {

static op_forward_func_t sigmoid_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		sigmoid_forward(input[0])
	);
	ctx.save_for_backward(result);	// Save the output tensor for backward
	return {result};
};

static op_backward_func_t sigmoid_backward_func = [](const std::vector<Tensor> &output_grad, OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor result = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		sigmoid_backward(output_grad[0], ctx.get_saved_tensors()[0])
	);
	return {result};
};

Tensor sigmoid(const Tensor &input) {
	return perform_op(sigmoid_forward_func, sigmoid_backward_func, {input})[0];
}

}
