#include "relu.h"

#include "src/backend/cpu/relu.h"
#include "src/backend/cuda/relu.h"

namespace NeuroFrame {

static op_forward_func_t relu_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	ctx.save_for_backward(input[0]);	// Save the input tensor for backward pass
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		relu_forward(input[0])
	);
	return {result};
};

static op_backward_func_t relu_backward_func = [](const std::vector<Tensor> &output_grad, OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor result = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		relu_backward(output_grad[0], ctx.get_saved_tensors()[0])
	);
	return {result};
};

Tensor relu(const Tensor &input) {
	return perform_op(relu_forward_func, relu_backward_func, {input})[0];
}

}
