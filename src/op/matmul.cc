#include "matmul.h"

#include <cassert>

#include "src/utils/utils.h"
#include "src/op/transpose.h"
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
static op_forward_func_t matmul_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	assert(other_args);

	do_basic_checkings_in_forward_and_backward(input, ctx);
	Tensor a = input[0];
	Tensor b = input[1];
	if ((a.shape.size() != 2 && a.shape.size() != 3) || 
		(b.shape.size() != 2 && b.shape.size() != 3)) {
		LOG_FATAL("matmul_forward_func: input tensors must be 2D or 3D "
				  "(dimensionality = %lu, %lu)", a.shape.size(), b.shape.size());
	}

	bool is_a_batched = a.shape.size() == 3;
	bool is_b_batched = b.shape.size() == 3;
	if (a.shape[(int)is_a_batched + 1] !=
		b.shape[(int)is_b_batched]) {
		LOG_FATAL("matmul_forward_func: input tensors' shapes are not compatible (inner dimensions do not match)"
				  "Shapes: %s, %s.",
				  vec_to_string(a.shape).c_str(), vec_to_string(b.shape).c_str());
	}
	if (is_a_batched && is_b_batched &&
		a.shape[0] != b.shape[0]) {
		LOG_FATAL("matmul_forward_func: input tensors' shapes are not compatible (batch sizes do not match) "
				   "Batch size: %ld vs %ld",
				   a.shape[0], b.shape[0]);
	}
	
	ctx.save_for_backward(a);
	ctx.save_for_backward(b);

	Tensor result = DISPATCH_TO_BACKEND(
		a.device.type,
		batched_matmul(a, b, false, false)
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

	Tensor a = ctx.get_saved_tensor(0);
	Tensor b = ctx.get_saved_tensor(1);
	bool is_a_batched = a.shape.size() == 3;
	bool is_b_batched = b.shape.size() == 3;

	if (is_a_batched != is_b_batched) {
		int64_t batch_size = is_a_batched ? a.shape[0] : b.shape[0];
		int64_t n = a.shape[is_a_batched ? 1 : 0];
		int64_t k = a.shape[is_a_batched ? 2 : 1];
		int64_t m = b.shape[is_b_batched ? 2 : 1];
		if (is_a_batched) {
			// Ci = Ai B
			Tensor a_grad = [&]() {
				return DISPATCH_TO_BACKEND(
					output_grad[0].device.type,
					batched_matmul(output_grad[0], b, false, true);
				);
			}();
			Tensor b_grad = [&]() {
				return DISPATCH_TO_BACKEND(
					output_grad[0].device.type,
					batched_matmul(
						a.reshape({batch_size*n, k}),
						output_grad[0].reshape({batch_size*n, m}),
						true,
						false
					);
				);
			}();
			return {a_grad, b_grad};
		} else {
			// Ci = A Bi
			Tensor b_grad = [&]() {
				return DISPATCH_TO_BACKEND(
					output_grad[0].device.type,
					batched_matmul(a, output_grad[0], true, false);
				);
			}();
			OpContext temp_ctx1, temp_ctx2;
			Tensor c_grad = transpose_forward_manual(		// c_grad: (batch_size, n, m)
				output_grad[0], temp_ctx1, 1, 2	// c_grad: (batch_size, m, n)
			).reshape({batch_size*m, n});		// c_grad: (batch_size*m, n)
			b = transpose_forward_manual(	// b: (batch_size, k, m)
				b, temp_ctx2, 1, 2	  // b: (batch_size, m, k)
			).reshape({batch_size*m, k});	// b: (batch_size*m, k)
			Tensor a_grad = [&]() {
				return DISPATCH_TO_BACKEND(
					output_grad[0].device.type,
					batched_matmul(c_grad, b, true, false);
				);
			}();
			return {a_grad, b_grad};
		}
	} else {
		Tensor a_grad = [&]() {
			return DISPATCH_TO_BACKEND(
				output_grad[0].device.type,
				batched_matmul(output_grad[0], b, false, true);
			);
		}();
		Tensor b_grad = [&]() {
			return DISPATCH_TO_BACKEND(
				output_grad[0].device.type,
				batched_matmul(a, output_grad[0], true, false);
			);
		}();
		return {a_grad, b_grad};
	}
};

Tensor matmul_forward_manual(const Tensor &a, const Tensor &b, OpContext &ctx) {
	return matmul_forward_func({a, b}, ctx, nullptr)[0];
}

std::vector<Tensor> matmul_backward_manual(const Tensor &output_grad, const OpContext &ctx) {
	return matmul_backward_func({output_grad}, ctx);
}

Tensor matmul(const Tensor &a, const Tensor &b) {
	return perform_op(matmul_forward_func, matmul_backward_func, {a, b}, nullptr)[0];
}

}
