#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CPU {

std::pair<Tensor, Tensor> batched_softmax_cross_entropy_loss_forward(const Tensor& answer, const Tensor& ground_truth);

Tensor batched_softmax_cross_entropy_loss_backward(const Tensor &output_grad, const Tensor &ground_truth, const Tensor &softmax_result);

}
