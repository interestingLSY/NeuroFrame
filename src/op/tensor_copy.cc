#include "tensor_copy.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

// Tensor copy
// Input:
//	- a: The Tensor to be copied
// Output:
//	- result: The result Tensor, = a
// SavedContext: None
static op_forward_func_t tensor_copy_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor result = a.copy();
	return {result};
};

static op_backward_func_t tensor_copy_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor grad = output_grad[0];
	Tensor grad_a = grad.copy();
	return {grad_a};
};

Tensor tensor_copy_forward_manual(const Tensor &input, OpContext &ctx) {
	return tensor_copy_forward_func({input}, ctx, nullptr)[0];
}

Tensor tensor_copy(const Tensor &input) {
	return perform_op(tensor_copy_forward_func, tensor_copy_backward_func, {input})[0];
}

}