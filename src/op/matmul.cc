#include "matmul.h"

#include "src/backend/cpu/matmul.h"
#include "src/backend/cuda/matmul.h"

namespace NeuroFrame {

// matmul_forward_func: forward function for matmul
// Inputs:
//	- 0: Matrix A
//	- 1: Matrix B
// Outputs:
//	- 0: Matrix C = A * B
// Saved tensors:
//	- 0: Matrix A
//  - 1: Matrix B
static op_forward_func_t matmul_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(input, ctx);
	ctx.save_for_backward(input[0]);
	ctx.save_for_backward(input[1]);
	Tensor result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		matmul(input[0], input[1], false, false)
	);
	return {result};
};

// matmul_backward_func: backward function for matmul
// Inputs:
//	- 0: Output gradient
// Outputs:
//	- 0: Input A's gradient
//	- 1: Input B's gradient
// Formular:
//	- A_grad = output_grad * B^T
//	- B_grad = A^T * output_grad
static op_backward_func_t matmul_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	Tensor a_grad = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		matmul(output_grad[0], ctx.get_saved_tensors()[1], false, true);
	);
	Tensor b_grad = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		matmul(ctx.get_saved_tensors()[0], output_grad[0], true, false);
	);
	return {a_grad, b_grad};
};

Tensor matmul_forward_manual(const Tensor &a, const Tensor &b, OpContext &ctx) {
	return matmul_forward_func({a, b}, ctx)[0];
}

std::vector<Tensor> matmul_backward_manual(const Tensor &output_grad, const OpContext &ctx) {
	return matmul_backward_func({output_grad}, ctx);
}

Tensor matmul(const Tensor &a, const Tensor &b) {
	return perform_op(matmul_forward_func, matmul_backward_func, {a, b})[0];
}

}
