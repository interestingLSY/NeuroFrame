#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CPU {

int64_t get_correct_sample_count(const Tensor &output, const Tensor &ground_truth);

void sgd_grad_update(Tensor &weight, const Tensor &grad, Tensor &momentum, double learning_rate, double momentum_factor, double weight_decay);

}
