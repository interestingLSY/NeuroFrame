#include "tensor_unary_op.h"

#include "op.h"

#include "src/tensor/tensor.h"
#include "src/op/tensor_binary_op.h"

#include "src/backend/cpu/tensor_unary_op.h"
#include "src/backend/cuda/tensor_unary_op.h"

namespace NeuroFrame {

// tensor_negate_forward_func
// Input:
//	- a: The tensor to be negated
// Output:
//	- result: The result tensor, = -a
// SavedContext: None
static op_forward_func_t tensor_negate_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		tensor_negate(a)
	);
	return {result};
};

static op_backward_func_t tensor_negate_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor a_grad = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		tensor_negate(output_grad[0])
	);
	return {a_grad};
};

Tensor tensor_negate_forward_manual(const Tensor &a, OpContext &ctx) {
	return tensor_negate_forward_func({a}, ctx, nullptr)[0];
}

Tensor tensor_negate(const Tensor &a) {
	return perform_op(tensor_negate_forward_func, tensor_negate_backward_func, {a})[0];
}

// tensor_inv_forward_func
// Input:
// 	- a: The tensor to be inversed
// Output:
//  - result: The result tensor, = 1/a
// SavedContext:
//  - result: 1/a
static op_forward_func_t tensor_inv_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		tensor_inv(a)
	);
	ctx.save_for_backward(result);
	return {result};
};

static op_backward_func_t tensor_inv_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor c = ctx.get_saved_tensor(0);
	// grad = output_grad * (-1/a^2) = - output_grad * c^2
	OpContext temp_ctx;
	Tensor a_grad = tensor_negate_forward_manual(
		tensor_mul_forward_manual(
			output_grad[0],
			tensor_mul_forward_manual(c, c, temp_ctx),
			temp_ctx
		),
		temp_ctx
	);
	return {a_grad};
};

Tensor tensor_inv_forward_manual(const Tensor &a, OpContext &ctx) {
	return tensor_inv_forward_func({a}, ctx, nullptr)[0];
}

Tensor tensor_inv(const Tensor &a) {
	return perform_op(tensor_inv_forward_func, tensor_inv_backward_func, {a})[0];
}

// tensor_exp_forward_func
// Input:
// 	- a: The tensor to be exped
// Output:
//  - result: The result tensor, = exp(a)
// SavedContext:
//  - result: exp(a)
static op_forward_func_t tensor_exp_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		tensor_exp(a)
	);
	ctx.save_for_backward(result);
	return {result};
};

static op_backward_func_t tensor_exp_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor c = ctx.get_saved_tensor(0);
	// grad = output_grad * exp(a)
	OpContext temp_ctx;
	Tensor a_grad = tensor_mul_forward_manual(output_grad[0], c, temp_ctx);
	return {a_grad};
};

Tensor tensor_exp_forward_manual(const Tensor &a, OpContext &ctx) {
	return tensor_exp_forward_func({a}, ctx, nullptr)[0];
}

Tensor tensor_exp(const Tensor &a) {
	return perform_op(tensor_exp_forward_func, tensor_exp_backward_func, {a})[0];
}

// tensor_log_forward_func
// Input:
// 	- a: The tensor to be logged
// Output:
//  - result: The result tensor, = log(a)
// SavedContext:
//  - a
static op_forward_func_t tensor_log_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		tensor_log(a)
	);
	ctx.save_for_backward(a);
	return {result};
};

static op_backward_func_t tensor_log_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor a = ctx.get_saved_tensor(0);
	// grad = output_grad / a
	OpContext temp_ctx;
	Tensor a_grad = tensor_div_forward_manual(output_grad[0], a, temp_ctx);
	return {a_grad};
};

Tensor tensor_log_forward_manual(const Tensor &a, OpContext &ctx) {
	return tensor_log_forward_func({a}, ctx, nullptr)[0];
}

Tensor tensor_log(const Tensor &a) {
	return perform_op(tensor_log_forward_func, tensor_log_backward_func, {a})[0];
}

}