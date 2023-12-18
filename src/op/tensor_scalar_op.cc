#include "tensor_scalar_op.h"

#include <cassert>
#include "op.h"

#include "src/tensor/tensor.h"

#include "src/op/tensor_binary_op.h"
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
#define DEFINE_TENSOR_SCALAR_OP(OP_NAME, FORWARD_OP_NAME, BACKWARD_EXPR, CHECK_TENSOR_SCALAR_DTYPE, SAVE_INPUT) \
static op_forward_func_t tensor_##OP_NAME##_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> { \
	Tensor input_tensor = input[0]; \
	TensorScalarOpArgs args = *(TensorScalarOpArgs*)other_args; \
	Scalar scalar = args.scalar; \
	if (CHECK_TENSOR_SCALAR_DTYPE && input_tensor.dtype != scalar.dtype) { \
		LOG_FATAL("tensor_" #OP_NAME "_forward_func: The input tensor's dtype and the scalar's dtype do not match"); \
	} \
	if (SAVE_INPUT) { \
		ctx.save_for_backward(input_tensor); \
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
	__attribute__((unused)) OpContext temp_ctx1; \
	__attribute__((unused)) OpContext temp_ctx2; \
	__attribute__((unused)) OpContext temp_ctx3; \
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

DEFINE_TENSOR_SCALAR_OP(adds, tensor_adds, output_grad, true, false)

DEFINE_TENSOR_SCALAR_OP(subs, tensor_subs, output_grad, true, false)

DEFINE_TENSOR_SCALAR_OP(muls, tensor_muls, tensor_muls_forward_manual(output_grad, scalar, temp_ctx1), true, false)

DEFINE_TENSOR_SCALAR_OP(divs, tensor_divs, tensor_divs_forward_manual(output_grad, scalar, temp_ctx1), true, false)

DEFINE_TENSOR_SCALAR_OP(pows, tensor_pows, 
	tensor_mul_forward_manual(
		output_grad,
		tensor_muls_forward_manual(
			tensor_pows_forward_manual(
				ctx.get_saved_tensor(0),
				scalar.as_double()-1.0f, temp_ctx1
			),
			scalar, temp_ctx2
		), temp_ctx3
	)
, false, true)


}