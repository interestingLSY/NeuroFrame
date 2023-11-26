#include "matmul.h"

#include "src/backend/cpu/matmul.h"
#include "src/backend/cuda/matmul.h"

namespace NeuroFrame {

struct MatmulArgs {
	bool transpose_a;
	bool transpose_b;
};

// matmul_forward_func: forward function for matmul
// Inputs:
//	- 0: Matrix A
//	- 1: Matrix B
// Outputs:
//	- 0: Matrix C = A * B
// Saved tensors:
//	- 0: Matrix A
//  - 1: Matrix B
static op_forward_func_t matmul_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	MatmulArgs args = *(MatmulArgs*)other_args;
	bool transpose_a = args.transpose_a;
	bool transpose_b = args.transpose_b;

	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor b = input[1];
	if ((a.shape.size() != 2 && a.shape.size() != 3) || 
		(b.shape.size() != 2 && b.shape.size() != 3)) {
		LOG_FATAL("matmul_forward_func: input tensors must be 2D or 3D");
	}

	bool is_a_batched = a.shape.size() == 3;
	bool is_b_batched = b.shape.size() == 3;
	if (a.shape[(int)is_a_batched + (transpose_a ? 0 : 1)] !=
		b.shape[(int)is_b_batched + (transpose_b ? 1 : 0)]) {
		LOG_FATAL("matmul_forward_func: input tensors' shapes are not compatible (inner dimensions do not match)");
	}
	if (is_a_batched && is_b_batched &&
		a.shape[0] != b.shape[0]) {
		LOG_FATAL("matmul_forward_func: input tensors' shapes are not compatible (batch sizes do not match)");
	}
	
	ctx.save_for_backward(a);
	ctx.save_for_backward(b);
	ctx.save_args(other_args, sizeof(MatmulArgs));

	Tensor result = DISPATCH_TO_BACKEND(
		a.device.type,
		batched_matmul(a, b, transpose_a, transpose_b)
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

	Tensor a = ctx.get_saved_tensors()[0];
	Tensor b = ctx.get_saved_tensors()[1];
	MatmulArgs args = *(MatmulArgs*)ctx.get_saved_args();
	bool transpose_a = args.transpose_a;
	bool transpose_b = args.transpose_b;
	bool is_a_batched = a.shape.size() == 3;
	bool is_b_batched = b.shape.size() == 3;

	if (is_a_batched || is_b_batched) {
		LOG_FATAL("matmul_backward_func: batched matmul is not supported yet");
	}

	Tensor a_grad = [&]() {
		if (!transpose_a) {
			return DISPATCH_TO_BACKEND(
				output_grad[0].device.type,
				matmul(output_grad[0], b, false, transpose_b ? false : true);
			);
		} else {
			return DISPATCH_TO_BACKEND(
				output_grad[0].device.type,
				matmul(b, output_grad[0], transpose_b ? true : false, true);
			);
		}
	}();
	Tensor b_grad = [&]() {
		if (!transpose_b) {
			return DISPATCH_TO_BACKEND(
				output_grad[0].device.type,
				matmul(a, output_grad[0], transpose_a ? false : true, false);
			);
		} else {
			return DISPATCH_TO_BACKEND(
				output_grad[0].device.type,
				matmul(output_grad[0], a, true, transpose_a ? true : false);
			);
		}
	}();
	return {a_grad, b_grad};
};

Tensor matmul_forward_manual(const Tensor &a, const Tensor &b, bool transpose_a, bool transpose_b, OpContext &ctx) {
	MatmulArgs args = {transpose_a, transpose_b};
	return matmul_forward_func({a, b}, ctx, &args)[0];
}

std::vector<Tensor> matmul_backward_manual(const Tensor &output_grad, const OpContext &ctx) {
	return matmul_backward_func({output_grad}, ctx);
}

Tensor matmul(const Tensor &a, const Tensor &b, bool transpose_a, bool transpose_b) {
	MatmulArgs args = {transpose_a, transpose_b};
	return perform_op(matmul_forward_func, matmul_backward_func, {a, b}, &args)[0];
}

}
