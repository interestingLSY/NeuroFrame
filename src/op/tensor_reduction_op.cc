#include "tensor_reduction_op.h"

#include <cassert>

#include "op.h"

#include "src/backend/cpu/tensor_reduction_op.h"
#include "src/backend/cuda/tensor_reduction_op.h"

namespace NeuroFrame {

// Tensor reduction sum
// Input:
//	- a: The Tensor
// Output:
//	- result: The result Tensor
// SavedContext:
//	- a (Indeed we only need the shape of a)
// other_args:
// 	- axis: The axis to reduce

struct TensorReductionArgs {
	int axis;
};

static op_forward_func_t tensor_reduction_sum_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	assert(other_args);
	int axis = ((TensorReductionArgs*)other_args)->axis;
	Tensor a = input[0];
	if (axis < -1 || axis >= a.dim()) {
		LOG_FATAL("The axis to reduce must be in range [0, dim) or -1");
	}
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		tensor_reduction_sum(a, axis)
	);
	return {result};
};

static op_backward_func_t tensor_reduction_sum_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor a_grad = output_grad[0];
	Tensor b_grad = output_grad[0];
	LOG_FATAL("Not implemented");
	return {a_grad, b_grad};
};

Tensor tensor_reduction_sum_forward_manual(const Tensor &a, OpContext &ctx, int axis) {
	TensorReductionArgs args;
	args.axis = axis;
	return tensor_reduction_sum_forward_func({a}, ctx, &args)[0];
}

Tensor tensor_reduction_sum_backward_manual(const Tensor &a, OpContext &ctx) {
	return tensor_reduction_sum_backward_func({a}, ctx)[0];
}

Tensor tensor_reduction_sum(const Tensor &a, int axis) {
	TensorReductionArgs args;
	args.axis = axis;
	return perform_op(tensor_reduction_sum_forward_func, tensor_reduction_sum_backward_func, {a}, &args)[0];
}

}
