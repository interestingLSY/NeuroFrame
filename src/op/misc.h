#pragma once

#include "op.h"

#include "src/tensor/tensor.h"

namespace NeuroFrame {

int64_t get_correct_sample_count(const Tensor &output, const Tensor &ground_truth);

void sgd_grad_update(Tensor &weight, const Tensor &grad, Tensor &momentum, double learning_rate, double momentum_factor, double weight_decay);

void adam_grad_update(Tensor &weight, const Tensor &grad, Tensor &momentum, Tensor &geo_mean, int64_t cur_timestamp, double learning_rate, double beta1, double beta2, double eps);

}