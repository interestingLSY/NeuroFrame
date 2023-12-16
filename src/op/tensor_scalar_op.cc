#include "tensor_scalar_op.h"

#include <cassert>
#include "op.h"

#include "src/tensor/tensor.h"

#include "src/backend/cpu/tensor_scalar_op.h"
#include "src/backend/cuda/tensor_scalar_op.h"

namespace NeuroFrame {
	
struct TensorScalarOpArgs {
	Scalar scalar;
};

// tensor_XXX_forward_func
// Input:
//	- a: The tensor
// Output:
//	- result: The result tensor, = a OP b
// other_args:
//	- b: The scalar
// Saved args:
//	- b: The scalar

// To avoid code duplication, we use macro to generate the code
// OP_NAME is the name of the operation, like `adds` or `muls`
// FORWARD_OP_NAME is the name of the forward function in the backend, like `tensor_adds` or `tensor_muls`
// BACKWARD_EXPR is the expression of the backward function, like `output_grad` or `tensor_muls_forward_manual(input_tensor, scalar, temp_ctx)`
#define DEFINE_TENSOR_SCALAR_OP(OP_NAME, FORWARD_OP_NAME, BACKWARD_EXPR) \
static op_forward_func_t tensor_##OP_NAME##_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> { \
	Tensor input_tensor = input[0]; \
	TensorScalarOpArgs args = *(TensorScalarOpArgs*)other_args; \
	Scalar scalar = args.scalar; \
	if (input_tensor.dtype != scalar.dtype) { \
		LOG_FATAL("tensor_" #OP_NAME "_forward_func: The input tensor's dtype and the scalar's dtype do not match"); \
	} \
	ctx.save_args(other_args, sizeof(TensorScalarOpArgs)); \
	Tensor result = DISPATCH_TO_BACKEND( \
		input[0].device.type, \
		FORWARD_OP_NAME(input_tensor, scalar) \
	); \
	return {result}; \
}; \
\
static op_backward_func_t tensor_##OP_NAME##_backward_func = [](const std::vector<Tensor> &output_grads, const OpContext &ctx) -> std::vector<Tensor> { \
	TensorScalarOpArgs args = *(TensorScalarOpArgs*)ctx.get_saved_args(); \
	Tensor output_grad = output_grads[0]; \
	__attribute__((unused)) Scalar scalar = args.scalar; \
	OpContext temp_ctx; \
	Tensor a_grad = BACKWARD_EXPR; \
	return {a_grad}; \
}; \
\
Tensor tensor_##OP_NAME##_forward_manual(const Tensor &a, const Scalar &b, OpContext &ctx) { \
	TensorScalarOpArgs args = {b}; \
	return tensor_##OP_NAME##_forward_func({a}, ctx, &args)[0]; \
} \
\
Tensor tensor_##OP_NAME##_backward_manual(const Tensor &a, OpContext &ctx) { \
	return tensor_##OP_NAME##_backward_func({a}, ctx)[0]; \
} \
\
Tensor tensor_##OP_NAME(const Tensor &a, const Scalar &b) { \
	TensorScalarOpArgs args = {b}; \
	return perform_op(tensor_##OP_NAME##_forward_func, tensor_##OP_NAME##_backward_func, {a}, &args)[0]; \
}

DEFINE_TENSOR_SCALAR_OP(adds, tensor_adds, output_grad)

DEFINE_TENSOR_SCALAR_OP(subs, tensor_subs, output_grad)

DEFINE_TENSOR_SCALAR_OP(muls, tensor_muls, tensor_muls_forward_manual(output_grad, scalar, temp_ctx))

DEFINE_TENSOR_SCALAR_OP(divs, tensor_divs, tensor_divs_forward_manual(output_grad, scalar, temp_ctx))

DEFINE_TENSOR_SCALAR_OP(pows, tensor_pows, tensor_muls_forward_manual(
	tensor_pows_forward_manual(
		output_grad,
		scalar.as_double()-1.0f, temp_ctx),
	scalar, temp_ctx
))


}