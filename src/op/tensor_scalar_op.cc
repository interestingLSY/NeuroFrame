#include "tensor_scalar_op.h"

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

// We define the following macros to avoid code duplication
// This macro will create a tensor-scalar op's forward function, backward function and entrypoint function 
#define DEFINE_TENSOR_SCALAR_OP_FUNCS(OP_NAME, FORWARD_OP_NAME, BACKWARD_OP) \
static op_forward_func_t tensor_##OP_NAME##_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> { \
	if (input.size() != 1) { \
		LOG_FATAL("tensor_" #OP_NAME "_forward_func: The number of input tensors should be 1"); \
	} \
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
};

DEFINE_TENSOR_SCALAR_OP_FUNCS(adds)

}