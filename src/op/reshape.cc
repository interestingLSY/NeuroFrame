#include "reshape.h"

#include <cassert>

namespace NeuroFrame {

// reshape: Tensor reshaping
// Input:
//	- input: The input Tensor
//	- shape: The new shape of the Tensor
// Output:
//	- result: The result Tensor
// SavedContext:
// 	- input (indeed we only need the shape of the input Tensor)
// other_args: None

static op_forward_func_t reshape_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	assert(other_args);
	std::vector<int64_t> shape = *(std::vector<int64_t>*)other_args;
	Tensor result = input[0].reshape(shape);
	ctx.save_for_backward(input[0]);
	return {result};
};

static op_backward_func_t reshape_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	std::vector<int64_t> old_shape = ctx.get_saved_tensor(0).shape;
	Tensor result = output_grad[0].reshape(old_shape);
	return {result};
};

Tensor reshape_forward_manual(const Tensor &input, const std::vector<int64_t> &shape, OpContext &ctx) {
	return reshape_forward_func({input}, ctx, (void*)&shape)[0];
}

Tensor reshape_backward_manual(const Tensor &output_grad, const std::vector<int64_t> &shape, const OpContext &ctx) {
	return reshape_backward_func({output_grad}, ctx)[0];
}

Tensor reshape(const Tensor &input, const std::vector<int64_t> &shape) {
	return perform_op(reshape_forward_func, reshape_backward_func, {input}, (void*)(&shape))[0];
}

}
