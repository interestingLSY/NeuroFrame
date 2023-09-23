#pragma once

#include "src/tensor/tensor.h"

namespace NeuroFrame::Backend::CUDA {

// ReLU: Rectified Linear Unit.
// ReLU(x) = max(0, x)
Tensor relu_forward(const Tensor &input);

// ReLU gradient.
// ReLU'(x) = 1 if x > 0 else 0
Tensor relu_backward(const Tensor &output_grad, const Tensor &input);

}