#include "batch_norm.h"

#include <cassert>

#include "src/basic/inference_mode.h"
#include "src/backend/cpu/batch_norm.h"
#include "src/backend/cuda/batch_norm.h"

namespace NeuroFrame {

// batch_norm: Batch normalization
// Input:
//	- input: The input Tensor
//	- gamma: The gamma Tensor
//	- beta: The beta Tensor
// Output:
//	- result: The result Tensor
// SavedContext:
//	- input: The input Tensor
//	- sample_mean
//	- sample_var
//	- gamma
// OtherArgs:
//	- momentum
//	- epsilon
//	- is_training
//	- running_mean
//	- running_var
// Here we pass "running mean" and "running var" in OtherArgs for convenience.

struct BatchNormArgs {
	double momentum;
	double epsilon;
	bool is_training;
	Tensor running_mean;
	Tensor running_var;
};

static op_forward_func_t batch_norm_forward_func = [](const std::vector<Tensor> &inputs, OpContext &ctx, void* other_args) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(inputs, ctx);
	assert(other_args);

	BatchNormArgs* args = (BatchNormArgs*)other_args;
	Tensor input = inputs[0];
	Tensor gamma = inputs[1];
	Tensor beta = inputs[2];

	if (args->running_mean.device != input.device ||
		args->running_var.device != input.device) {
		LOG_FATAL("batch_norm: running_mean and running_var must be on the same device as input");
	}
	if (args->running_mean.dtype != input.dtype ||
		args->running_var.dtype != input.dtype) {
		LOG_FATAL("batch_norm: running_mean and running_var must have the same dtype as input");
	}

	auto [output, sample_mean, sample_var] = DISPATCH_TO_BACKEND(
		input.device.type,
		batch_norm(
			input,
			gamma,
			beta,
			args->running_mean,
			args->running_var,
			args->momentum,
			args->epsilon,
			args->is_training
		)
	);

	ctx.save_for_backward(input);
	ctx.save_for_backward(sample_mean);
	ctx.save_for_backward(sample_var);
	ctx.save_for_backward(gamma);

	return {output};
};

static op_backward_func_t batch_norm_backward_func = [](const std::vector<Tensor> &output_grad, const OpContext &ctx) -> std::vector<Tensor> {
	do_basic_checkings_in_forward_and_backward(output_grad, ctx);

	Tensor input = ctx.get_saved_tensor(0);
	Tensor sample_mean = ctx.get_saved_tensor(1);
	Tensor sample_var = ctx.get_saved_tensor(2);
	Tensor gamma = ctx.get_saved_tensor(3);

	auto [input_grad, gamma_grad] = DISPATCH_TO_BACKEND(
		input.device.type,
		batch_norm_backward(
			output_grad[0],
			input,
			gamma,
			sample_mean,
			sample_var
		)
	);

	Tensor beta_grad = DISPATCH_TO_BACKEND(
		input.device.type,
		batch_norm_beta_grad(
			output_grad[0]
		)
	);

	return {input_grad, gamma_grad, beta_grad};
};

Tensor batch_norm(const Tensor &input, const Tensor &gamma, const Tensor &beta, const Tensor &mean, const Tensor &variance, double momentum, double epsilon) {
	bool is_training = !is_inference_mode();
	BatchNormArgs args = {momentum, epsilon, is_training, mean, variance};
	return perform_op(batch_norm_forward_func, batch_norm_backward_func, {input, gamma, beta}, &args)[0];
}

}
