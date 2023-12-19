#include "cross_entropy_loss.h"

#include "op.h"

#include "src/tensor/tensor.h"

#include "src/backend/cpu/cross_entropy_loss.h"
#include "src/backend/cuda/cross_entropy_loss.h"

namespace NeuroFrame {

// cross_entropy_loss_forward_func
// Input:
//  - input: the input tensor, (batch_size, num_classes)
//	- ground_truth: the ground truth, (batch_size)
// Output:
//  - result: the result, (batch_size)
// SavedContext:
//  - saved_tensors[0]: the softmax result, (batch_size, num_classes)
//  - saved_tensors[1]: the ground truth, (batch_size)
// OtherArgs: None

static op_forward_func_t cross_entropy_loss_forward_func = [](const std::vector<Tensor> &inputs, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	if (!is_on_same_device(inputs)) {
		LOG_FATAL("All the input tensors must be on the same device.");
	}
	
	Tensor input = inputs[0];
	Tensor ground_truth = inputs[1];
	if (input.dim() != 2) {
		LOG_FATAL("The input tensor must be 2D.");
	}
	if (ground_truth.dim() != 1) {
		LOG_FATAL("The ground truth tensor must be 1D.");
	}
	if (input.shape[0] != ground_truth.shape[0]) {
		LOG_FATAL("The batch size of input and ground truth must be the same.");
	}
	if (ground_truth.dtype != dtype_t::INT32) {
		LOG_FATAL("The dtype of ground truth must be INT32.");
	}

	auto [result, softmax_result] = DISPATCH_TO_BACKEND(
		input.device.type,
		batched_softmax_cross_entropy_loss_forward(input, ground_truth)
	);

	ctx.save_for_backward(softmax_result);
	ctx.save_for_backward(ground_truth);

	return {result};
};

static op_backward_func_t cross_entropy_loss_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	Tensor softmax_result = ctx.get_saved_tensors()[0];
	Tensor ground_truth = ctx.get_saved_tensors()[1];

	Tensor input_grad = DISPATCH_TO_BACKEND(
		output_grad[0].device.type,
		batched_softmax_cross_entropy_loss_backward(output_grad[0], ground_truth, softmax_result)
	);

	return {input_grad};
};

Tensor cross_entropy_loss_forward_manual(const Tensor &input, const Tensor &ground_truth, OpContext &ctx) {
	return cross_entropy_loss_forward_func({input, ground_truth}, ctx, nullptr)[0];
}

Tensor cross_entropy_loss_backward_manual(const Tensor &output_grad, const OpContext &ctx) {
	return cross_entropy_loss_backward_func({output_grad}, ctx)[0];
}

Tensor cross_entropy_loss(const Tensor &input, const Tensor &ground_truth) {
	return perform_op(cross_entropy_loss_forward_func, cross_entropy_loss_backward_func, {input, ground_truth})[0];
}

}