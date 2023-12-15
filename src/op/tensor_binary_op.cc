#include "tensor_binary_op.h"
#include "src/op/tensor_unary_op.h"

#include "src/backend/cpu/tensor_binary_op.h"
#include "src/backend/cuda/tensor_binary_op.h"

namespace NeuroFrame {

// Tensor addition
// Input:
//	- a: The first Tensor
//	- b: The second Tensor
// Output:
//	- result: The result Tensor, = a+b
// SavedContext: None
static op_forward_func_t tensor_add_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor b = input[1];
	if (a.shape != b.shape) {
		LOG_FATAL("The shape of two input tensors are not the same");
	}
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		tensor_add(a, b)
	);
	return {result};
};

static op_backward_func_t tensor_add_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor a_grad = output_grad[0];
	Tensor b_grad = output_grad[0];
	return {a_grad, b_grad};
};

Tensor tensor_add_forward_manual(const Tensor &a, const Tensor &b, OpContext &ctx) {
	return tensor_add_forward_func({a, b}, ctx, nullptr)[0];
}

Tensor tensor_add_backward_manual(const Tensor &a, const Tensor &b, OpContext &ctx) {
	return tensor_add_backward_func({a, b}, ctx)[0];
}

Tensor tensor_add(const Tensor &a, const Tensor &b) {
	return perform_op(tensor_add_forward_func, tensor_add_backward_func, {a, b})[0];
}

// Tensor subtraction
// Input:
//	- a: The first Tensor
//	- b: The second Tensor
// Output:
//	- result: The result Tensor, = a-b
// SavedContext: None
static op_forward_func_t tensor_sub_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor b = input[1];
	if (a.shape != b.shape) {
		LOG_FATAL("The shape of two input tensors are not the same");
	}
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		tensor_sub(a, b)
	);
	return {result};
};

static op_backward_func_t tensor_sub_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor a_grad = output_grad[0];
	OpContext temp_ctx;
	Tensor b_grad = tensor_negate_forward_manual(output_grad[0], temp_ctx);
	return {a_grad, b_grad};
};

Tensor tensor_sub_forward_manual(const Tensor &a, const Tensor &b, OpContext &ctx) {
	return tensor_sub_forward_func({a, b}, ctx, nullptr)[0];
}

Tensor tensor_sub_backward_manual(const Tensor &a, const Tensor &b, OpContext &ctx) {
	return tensor_sub_backward_func({a, b}, ctx)[0];
}

Tensor tensor_sub(const Tensor &a, const Tensor &b) {
	return perform_op(tensor_sub_forward_func, tensor_sub_backward_func, {a, b})[0];
}

// Tensor multiply
// Input:
//	- a: The first Tensor
//	- b: The second Tensor
// Output:
//	- result: The result Tensor, = a*b
// SavedContext: 
// - a
// - b
static op_forward_func_t tensor_mul_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor b = input[1];
	if (a.shape != b.shape) {
		LOG_FATAL("The shape of two input tensors are not the same");
	}
	ctx.save_for_backward(a);
	ctx.save_for_backward(b);
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		tensor_mul(a, b)
	);
	return {result};
};

static op_backward_func_t tensor_mul_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor a = ctx.get_saved_tensors()[0];
	Tensor b = ctx.get_saved_tensors()[1];
	OpContext temp_ctx;
	Tensor a_grad = tensor_mul(b, output_grad[0]);
	Tensor b_grad = tensor_mul(a, output_grad[0]);
	return {a_grad, b_grad};
};

Tensor tensor_mul_forward_manual(const Tensor &a, const Tensor &b, OpContext &ctx) {
	return tensor_mul_forward_func({a, b}, ctx, nullptr)[0];
}

Tensor tensor_mul_backward_manual(const Tensor &a, const Tensor &b, OpContext &ctx) {
	return tensor_mul_backward_func({a, b}, ctx)[0];
}

Tensor tensor_mul(const Tensor &a, const Tensor &b) {
	return perform_op(tensor_mul_forward_func, tensor_mul_backward_func, {a, b})[0];
}

// Tensor division
// Input:
//	- a: The first Tensor
//	- b: The second Tensor
// Output:
//	- result: The result Tensor, = a/b
// SavedContext: 
// - a
// - b
static op_forward_func_t tensor_div_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor b = input[1];
	if (a.shape != b.shape) {
		LOG_FATAL("The shape of two input tensors are not the same");
	}
	ctx.save_for_backward(a);
	ctx.save_for_backward(b);
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		tensor_div(a, b)
	);
	return {result};
};

static op_backward_func_t tensor_div_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor a = ctx.get_saved_tensors()[0];
	Tensor b = ctx.get_saved_tensors()[1];
	OpContext temp_ctx;
	Tensor a_grad = tensor_div_forward_manual(output_grad[0], b, temp_ctx);
	Tensor b_grad = tensor_div_forward_manual(
		tensor_negate_forward_manual(
			tensor_mul_forward_manual(
				a,
				output_grad[0], temp_ctx
			), temp_ctx
		),
		tensor_mul_forward_manual(b, b, temp_ctx), temp_ctx
	);
	return {a_grad, b_grad};
};

Tensor tensor_div_forward_manual(const Tensor &a, const Tensor &b, OpContext &ctx) {
	return tensor_div_forward_func({a, b}, ctx, nullptr)[0];
}

Tensor tensor_div_backward_manual(const Tensor &a, const Tensor &b, OpContext &ctx) {
	return tensor_div_backward_func({a, b}, ctx)[0];
}

Tensor tensor_div(const Tensor &a, const Tensor &b) {
	return perform_op(tensor_div_forward_func, tensor_div_backward_func, {a, b})[0];
}


}
