#include "tensor_reduction_op.h"

#include <cassert>

#include "op.h"

#include "src/backend/cpu/tensor_reduction_op.h"
#include "src/backend/cuda/tensor_reduction_op.h"
#include "src/op/broadcast.h"
#include "src/op/reshape.h"

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
	ctx.save_for_backward(a);
	ctx.save_args(other_args, sizeof(TensorReductionArgs));
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		tensor_reduction_sum(a, axis)
	);
	return {result};
};

static op_backward_func_t tensor_reduction_sum_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);

	OpContext temp_ctx;
	Tensor grad = output_grad[0];
	int axis = ((TensorReductionArgs*)ctx.get_saved_args())->axis;
	if (axis != -1) {
		// If axis is not -1, we need to first reshape the grad to the same dimension as the input
		// (unsqueeze the grad along the axis)
		std::vector<int64_t> unsqueezed_shape = grad.shape;
		unsqueezed_shape.insert(unsqueezed_shape.begin() + axis, 1);
		grad = reshape_forward_manual(grad, temp_ctx, unsqueezed_shape);
	}

	std::vector<int64_t> old_shape = ctx.get_saved_tensor(0).shape;
	Tensor input_grad = broadcast_to_forward_manual(
		grad,
		temp_ctx,
		old_shape
	);
	return {input_grad};
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
