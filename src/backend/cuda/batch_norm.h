#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

std::tuple<Tensor, Tensor, Tensor> batch_norm(
	const Tensor &input,
	const Tensor &gamma,
	const Tensor &beta,
	Tensor &running_mean,
	Tensor &running_var,
	double momentum,
	double eps,
	bool is_training
);

std::tuple<Tensor, Tensor> batch_norm_backward(
	const Tensor &output_grad,
	const Tensor &input,
	const Tensor &gamma,
	const Tensor &sample_mean,
	const Tensor &sample_var
);

Tensor batch_norm_beta_grad(
	const Tensor &output_grad
);

}
