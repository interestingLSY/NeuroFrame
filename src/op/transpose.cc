#include "transpose.h"

#include <cassert>

#include "src/backend/cpu/transpose.h"
#include "src/backend/cuda/transpose.h"

namespace NeuroFrame {

// transpose_forward_func: forward function for transpose
// Inputs:
//	- 0: Matrix A
// Outputs:
//	- 0: Matrix B = A^T
// Saved tensors: None
// other_args (8 byte)
// 	- 0: int axe1
// 	- 1: int axe2

struct TransposeForwardArgs {
	int axe1;
	int axe2;
};

static op_forward_func_t transpose_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	
	assert(other_args);
	TransposeForwardArgs args = *(TransposeForwardArgs*)other_args;
	
	Tensor a = input[0];
	int dim = a.dim();
	if (args.axe1 < 0 || args.axe1 >= dim || args.axe2 < 0 || args.axe2 >= dim) {
		LOG_FATAL("Transpose: Invalid axes");
	}

	Tensor result = DISPATCH_TO_BACKEND(
		a.device.type,
		transpose(a, args.axe1, args.axe2)
	);
	ctx.save_args(other_args, sizeof(TransposeForwardArgs));

	return {result};
};

static op_backward_func_t transpose_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);

	TransposeForwardArgs args = *(TransposeForwardArgs*)ctx.get_saved_args();

	Tensor result = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		transpose(output_grad[0], args.axe1, args.axe2)
	);

	return {result};
};

Tensor transpose_forward_manual(const Tensor &input, OpContext &ctx, int axe1, int axe2) {
	TransposeForwardArgs args = {axe1, axe2};
	return transpose_forward_func({input}, ctx, &args)[0];
}

Tensor transpose_backward_manual(const Tensor &output_grad, const OpContext &ctx) {
	return transpose_backward_func({output_grad}, ctx)[0];
}

Tensor transpose(const Tensor &input, int axe1, int axe2) {
	TransposeForwardArgs args = {axe1, axe2};
	return perform_op(transpose_forward_func, transpose_backward_func, {input}, &args)[0];
}

}
