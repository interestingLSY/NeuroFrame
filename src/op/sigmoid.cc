#include "sigmoid.h"

#include "src/backend/cpu/sigmoid.h"
#include "src/backend/cuda/sigmoid.h"

namespace NeuroFrame {

// sigmoid: The sigmoid activation function
// Input:
//	- input: The input tensor, (N)
// Output:
//	- result: The result tensor, (N)
// SavedContext:
//	- saved_tensors[0]: The output tensor
// OtherArgs: None
static op_forward_func_t sigmoid_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		sigmoid_forward(input[0])
	);
	ctx.save_for_backward(result);	// Save the output tensor for backward
	return {result};
};

static op_backward_func_t sigmoid_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor result = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		sigmoid_backward(output_grad[0], ctx.get_saved_tensors()[0])
	);
	return {result};
};

Tensor sigmoid_forward_manual(const Tensor &input, OpContext &ctx) {
	return sigmoid_forward_func({input}, ctx, nullptr)[0];
}

Tensor sigmoid_backward_manual(const Tensor &output_grad, const OpContext &ctx) {
	return sigmoid_backward_func({output_grad}, ctx)[0];
}

Tensor sigmoid(const Tensor &input) {
	return perform_op(sigmoid_forward_func, sigmoid_backward_func, {input})[0];
}

}
