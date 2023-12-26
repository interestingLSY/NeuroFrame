#include "misc.h"

#include "src/backend/cpu/misc.h"
#include "src/backend/cuda/misc.h"

namespace NeuroFrame {

// get_correct_sample_count: Get the number of correct samples in a batch
// Input:
//	- output: The output tensor, (batch_size, num_classes)
//	- ground_truth: The ground truth tensor, (batch_size)
// Output:
//	- result: The number of correct samples in the batch
// SavedContext: None
// OtherArgs: None
static op_forward_func_t get_correct_sample_count_forward_func = [](const std::vector<Tensor> &input, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	Tensor pred_output = input[0];
	Tensor ground_truth = input[1];

	if (pred_output.shape.size() != 2) {
		LOG_FATAL("The shape of the prediction output tensor must be (batch_size, num_classes)");
	}
	if (ground_truth.shape.size() != 1) {
		LOG_FATAL("The shape of the ground truth tensor must be (batch_size)");
	}
	if (pred_output.shape[0] != ground_truth.shape[0]) {
		LOG_FATAL("The batch size of the prediction output tensor and the ground truth tensor must be the same");
	}
	if (ground_truth.dtype != dtype_t::INT32) {
		LOG_FATAL("The dtype of the ground truth tensor must be INT32");
	}

	int64_t result = DISPATCH_TO_BACKEND(
		input[0].device.type,
		get_correct_sample_count(input[0], input[1])
	);
	Tensor result_tensor = Tensor::fill(result, {}, dtype_t::INT64, Device::cpu());
	return {result_tensor};
};

static op_backward_func_t get_correct_sample_count_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);
	LOG_FATAL("get_correct_sample_count_backward is not implemented");
	return {};
};

int64_t get_correct_sample_count(const Tensor &pred_output, const Tensor &ground_truth) {
	return perform_op(get_correct_sample_count_forward_func, get_correct_sample_count_backward_func, {pred_output, ground_truth})[0].as_scalar().as_int64();
}

// sgd_grad_update
void sgd_grad_update(Tensor &weight, const Tensor &grad, Tensor &momentum, double learning_rate, double momentum_factor, double weight_decay) {
	DISPATCH_TO_BACKEND(
		weight.device.type,
		sgd_grad_update(weight, grad, momentum, learning_rate, momentum_factor, weight_decay)
	);
}

}
