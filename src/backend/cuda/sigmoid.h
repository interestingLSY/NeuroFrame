#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

Tensor sigmoid_forward(const Tensor &input);

// ReLU gradient.
// ReLU'(x) = 1 if x > 0 else 0
Tensor sigmoid_backward(const Tensor &output_grad, const Tensor &output);

}